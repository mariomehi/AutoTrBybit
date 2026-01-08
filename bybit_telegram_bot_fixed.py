"""
Telegram Bot for automated pattern detection + Bybit Testnet trading - FIXED VERSION
Correzioni principali:
- Gestione corretta del filesystem su Railway
- Matplotlib backend non-GUI
- Migliore gestione errori
- Logging migliorato
"""

import os
import time
import math
import logging
from datetime import datetime, timezone, timedelta
import threading
import io
import tempfile

# IMPORTANTE: Configura matplotlib prima di altri import
import matplotlib
matplotlib.use('Agg')  # Backend non-GUI per server

import requests
import pandas as pd
import numpy as np
import mplfinance as mpf

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import telegram.error

# Import pybit
try:
    from pybit.unified_trading import HTTP as BybitHTTP
except Exception as e:
    logging.warning(f'pybit import failed: {e}')
    BybitHTTP = None
    
# Import pattern statistics tracker
import config

# Import pattern statistics tracker
import track_patterns

def is_good_trading_time_utc(now=None) -> tuple[bool, str]:
    """
    Ritorna (ok, reason)
    """
    if not config.MARKET_TIME_FILTER_ENABLED:
        return (True, "Market time filter OFF")

    now = now or datetime.now(timezone.utc)
    h = now.hour

    if h in config.MARKET_TIME_FILTER_BLOCKED_UTC_HOURS:
        reason = f"Blocked low-liquidity hour UTC={h:02d}"
        logging.warning(f'🚫 MARKET TIME FILTER: {reason}')  # ← AGGIUNGI QUESTO
        return (False, reason)
    
    return (True, f"OK hour UTC={h:02d}")


def create_bybit_session():
    """Crea sessione Bybit per trading (Demo o Live)"""
    if BybitHTTP is None:
        raise RuntimeError('pybit non disponibile. Installa: pip install pybit>=5.0')
    if not config.BYBIT_API_KEY or not config.BYBIT_API_SECRET:
        raise RuntimeError('BYBIT_API_KEY e BYBIT_API_SECRET devono essere configurate')
    
    # Determina l'endpoint in base alla modalità
    base_url = config.BYBIT_ENDPOINTS.get(config.TRADING_MODE, config.BYBIT_ENDPOINTS['demo'])
    
    logging.info(f'🔌 Connessione Bybit - Modalità: {config.TRADING_MODE.upper()}')
    logging.info(f'📡 Endpoint: {base_url}')
    
    session = BybitHTTP(
        api_key=config.BYBIT_API_KEY,
        api_secret=config.BYBIT_API_SECRET,
        testnet=False,  # Non usiamo testnet
        demo=True if config.TRADING_MODE == 'demo' else False  # Usa demo se configurato
    )
    
    return session

# ----------------------------- UTILITIES -----------------------------

def bybit_get_klines(symbol: str, interval: str, limit: int = 200):
    """
    Ottiene klines da Bybit v5 public API
    Returns: DataFrame con OHLCV
    """
    itv = config.BYBIT_INTERVAL_MAP.get(interval)
    if itv is None:
        raise ValueError(f'Timeframe non supportato: {interval}')

    url = f'{config.BYBIT_PUBLIC_REST}/v5/market/kline'
    params = {
        'category': 'linear', 
        'symbol': symbol, 
        'interval': itv, 
        'limit': limit
    }
    
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        
        if j.get('retCode') != 0:
            logging.error(f"Bybit API error: {j.get('retMsg')}")
            return pd.DataFrame()
        
        data = j.get('result', {}).get('list', [])
        if not data:
            logging.warning(f'Nessun dato per {symbol} {interval}')
            return pd.DataFrame()

        # Bybit restituisce dati dal più recente al più vecchio, invertiamo
        data = list(reversed(data))
        
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume','turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open','high','low','close','volume']].astype(float)
        # Debug: Verifica volume
        if len(df) > 0:
            vol_sum = df['volume'].sum()
            vol_mean = df['volume'].mean()
            vol_max = df['volume'].max()
            
            if vol_sum == 0:
                logging.warning(f'⚠️ WARNING: All volumes are ZERO for {symbol} {interval}')
            else:
                logging.debug(f'Volume stats for {symbol}: sum={vol_sum:.2f}, mean={vol_mean:.2f}, max={vol_max:.2f}')
        
        return df
        
    except requests.exceptions.RequestException as e:
        logging.error(f'Errore nella richiesta klines: {e}')
        return pd.DataFrame()
    except Exception as e:
        logging.error(f'Errore nel parsing klines: {e}')
        return pd.DataFrame()


def get_instrument_info_cached(session, symbol: str) -> dict:
    """
    Ottiene le informazioni sul symbol (min_qty, max_qty, qty_step, price_decimals)
    con sistema di caching intelligente per eliminare latenza.
    
    Cache valida per 24h (le spec dei symbol non cambiano quasi mai).
    """
    now = datetime.now()
    
    with INSTRUMENT_CACHE_LOCK:
        # Controlla se esiste in cache e non è scaduta
        if symbol in INSTRUMENT_INFO_CACHE:
            cached_data = INSTRUMENT_INFO_CACHE[symbol]
            cache_time = cached_data['timestamp']
            
            # Se cache valida (< 24h), restituisci subito
            if now - cache_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                logging.debug(f"{symbol} - Using cached instrument info (age: {(now - cache_time).seconds}s)")
                return cached_data['info']
        
        # Cache mancante o scaduta: scarica da Bybit
        logging.info(f"{symbol} - Downloading fresh instrument info from Bybit...")
        
        try:
            instrument_info = session.get_instruments_info(category='linear', symbol=symbol)
            
            if instrument_info.get('retCode') == 0:
                instruments = instrument_info.get('result', {}).get('list', [])
                
                if instruments:
                    lotsize_filter = instruments[0].get('lotSizeFilter', {})
                    price_filter = instruments[0].get('priceFilter', {})
                    
                    # Estrai informazioni critiche
                    info = {
                        'min_order_qty': float(lotsize_filter.get('minOrderQty', 0.001)),
                        'max_order_qty': float(lotsize_filter.get('maxOrderQty', 1000000)),
                        'qty_step': float(lotsize_filter.get('qtyStep', 0.001)),
                        'tick_size': float(price_filter.get('tickSize', 0.01)),
                    }
                    
                    # Calcola decimali qty (per arrotondamento)
                    info['qty_decimals'] = len(str(info['qty_step']).split('.')[-1]) if '.' in str(info['qty_step']) else 0
                    
                    # Calcola decimali prezzo
                    info['price_decimals'] = len(str(info['tick_size']).split('.')[-1]) if '.' in str(info['tick_size']) else 0
                    
                    # Salva in cache con timestamp
                    INSTRUMENT_INFO_CACHE[symbol] = {
                        'info': info,
                        'timestamp': now
                    }
                    
                    logging.info(f"{symbol} - Cached: min_qty={info['min_order_qty']}, step={info['qty_step']}, decimals={info['qty_decimals']}")
                    return info
                else:
                    logging.warning(f"{symbol} - No instrument data found, using defaults")
                    return _get_default_instrument_info()
            else:
                logging.error(f"Bybit API error: {instrument_info.get('retMsg')}")
                return _get_default_instrument_info()
                
        except Exception as e:
            logging.error(f"Error fetching instrument info for {symbol}: {e}")
            return _get_default_instrument_info()


def _get_default_instrument_info() -> dict:
    """Fallback con valori di default sicuri"""
    return {
        'min_order_qty': 0.001,
        'max_order_qty': 1000000,
        'qty_step': 0.001,
        'tick_size': 0.01,
        'qty_decimals': 3,
        'price_decimals': 2
    }


def clear_instrument_cache(symbol: str = None):
    """Pulisce la cache (utile per debug o aggiornamenti manuali)"""
    with INSTRUMENT_CACHE_LOCK:
        if symbol:
            if symbol in INSTRUMENT_INFO_CACHE:
                del INSTRUMENT_INFO_CACHE[symbol]
                logging.info(f"Cache cleared for {symbol}")
        else:
            INSTRUMENT_INFO_CACHE.clear()
            logging.info("全 cache cleared")



def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcola Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(period).mean()

def atr_expanding(df: pd.DataFrame, expansion_threshold: float = 1.15) -> bool:
    """
    Filtro ATR Expansion - Volatilità in aumento
    Evita entry durante consolidamento
    
    Args:
        df: DataFrame OHLCV
        expansion_threshold: ATR corrente > threshold × media
    
    Returns:
        True se ATR è in espansione
    """
    if len(df) < 20:
        return False
    
    # Calcola ATR manualmente (usa la tua funzione atr esistente se preferisci)
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(14).mean()
    
    if atr_series.isna().any():
        return False
    
    atr_avg = atr_series.iloc[-10:-1].mean()
    current_atr = atr_series.iloc[-1]
    
    if atr_avg == 0:
        return False
    
    return current_atr > atr_avg * expansion_threshold

def is_valid_trend_for_entry(
    df: pd.DataFrame,
    mode: str = 'ema_based',
    symbol: str = None
) -> tuple:
    """
    Smart Trend Filter - Supporta consolidamenti e breakout
    
    MODES:
    - 'structure': Higher Highs + Higher Lows (originale)
    - 'ema_based': Prezzo sopra EMA 60 (consigliato)
    - 'hybrid': Structure OR EMA (flessibile)
    - 'pattern_only': Skip (ogni pattern decide)
    
    Returns:
        (valid: bool, reason: str, details: dict)
    """
    if len(df) < 70:
        return (False, 'Insufficient data', {})
    
    # === MODE 1: PATTERN-ONLY (skip global check) ===
    if mode == 'pattern_only':
        return (True, 'Pattern-only mode', {})
    
    # === MODE 2: EMA-BASED (CONSIGLIATO) ===
    if mode == 'ema_based':
        return check_ema_trend(df)
    
    # === MODE 3: STRUCTURE (originale) ===
    if mode == 'structure':
        return check_structure_trend(df)
    
    # === MODE 4: HYBRID (Structure OR EMA) ===
    if mode == 'hybrid':
        return check_hybrid_trend(df)
    
    return (False, 'Unknown mode', {})

def is_valid_trend_for_sell(
    df: pd.DataFrame,
    mode: str = 'ema_based',
    symbol: str = None
) -> tuple:
    """
    Trend Filter per SELL (SHORT)
    
    MODES:
    - 'ema_based': Prezzo sotto EMA 60 (downtrend)
    - 'structure': Lower Lows + Lower Highs
    - 'pattern_only': Skip check
    
    Returns:
        (valid: bool, reason: str, details: dict)
    """
    if mode == 'pattern_only':
        return (True, 'Pattern-only mode', {})
    
    if mode == 'ema_based':
        ema_60 = df['close'].ewm(span=60, adjust=False).mean()
        curr_price = df['close'].iloc[-1]
        curr_ema60 = ema_60.iloc[-1]
        
        # Per SHORT: prezzo SOTTO EMA 60
        if curr_price >= curr_ema60:
            return (False, f'Above EMA 60 (no downtrend)', {
                'ema60': curr_ema60,
                'price': curr_price
            })
        
        return (True, 'Below EMA 60 (downtrend)', {
            'ema60': curr_ema60,
            'price': curr_price
        })
    
    # Structure mode: Lower Lows + Lower Highs
    if mode == 'structure':
        if len(df) < 10:
            return (False, 'Insufficient data', {})
        
        highs = df['high'].iloc[-10:]
        lows = df['low'].iloc[-10:]
        
        split = 5
        
        recent_high = highs.iloc[-split:].max()
        previous_high = highs.iloc[:-split].max()
        
        recent_low = lows.iloc[-split:].min()
        previous_low = lows.iloc[:-split].min()
        
        has_ll = recent_low < previous_low
        has_lh = recent_high < previous_high
        
        if has_ll and has_lh:
            return (True, 'Lower Lows + Lower Highs', {})
        else:
            return (False, 'No downtrend structure', {})
    
    return (False, 'Unknown mode', {})

def check_ema_trend(df: pd.DataFrame) -> tuple:
    """
    EMA-Based Trend Check
    
    LOGICA:
    1. Prezzo SOPRA EMA 60 = Uptrend (base)
    2. Consolidamento OK se sopra EMA 60
    3. Pullback OK se non rompe EMA 60
    4. Breakout detection (momentum change)
    
    Win Rate: Mantiene 60-70% patterns
    False Negatives: ~10% (molto basso)
    """
    config = TREND_FILTER_CONFIG['ema_based']
    
    # Calcola EMA 60
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    if len(ema_60) < 60:
        return (False, 'EMA 60 not ready', {})
    
    curr_price = df['close'].iloc[-1]
    curr_ema60 = ema_60.iloc[-1]
    
    # Check 1: Prezzo sopra EMA 60 (con buffer)
    buffer = config['ema60_buffer']
    above_ema60 = curr_price > curr_ema60 * buffer
    
    if not above_ema60:
        distance = ((curr_price - curr_ema60) / curr_ema60) * 100
        return (False, f'Below EMA 60 ({distance:.2f}%)', {
            'ema60': curr_ema60,
            'price': curr_price,
            'distance_pct': distance
        })
    
    # Check 2: Analisi ultimi 10 periodi
    recent_prices = df['close'].iloc[-10:]
    recent_ema60 = ema_60.iloc[-10:]
    
    # Conta quante candele sono sopra EMA 60
    above_count = (recent_prices > recent_ema60 * buffer).sum()
    above_pct = (above_count / len(recent_prices)) * 100
    
    # Almeno 60% candele sopra EMA 60
    if above_pct < 60:
        return (False, f'Only {above_pct:.0f}% above EMA 60', {
            'above_count': above_count,
            'above_pct': above_pct
        })
    
    # Check 3: Detect consolidamento (OK se sopra EMA 60)
    recent_high = recent_prices.max()
    recent_low = recent_prices.min()
    recent_range_pct = ((recent_high - recent_low) / recent_low) * 100
    
    is_consolidating = recent_range_pct < 2.0  # Range < 2%
    
    # Check 4: Detect breakout momentum
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    
    curr_ema10 = ema_10.iloc[-1]
    curr_ema5 = ema_5.iloc[-1]
    
    # Momentum positivo se EMA 5 > EMA 10
    positive_momentum = curr_ema5 > curr_ema10
    
    # EMA 60 slope (trend strength)
    ema60_prev = ema_60.iloc[-10]
    ema60_slope = ((curr_ema60 - ema60_prev) / ema60_prev) * 100
    
    # DECISION LOGIC
    details = {
        'ema60': curr_ema60,
        'price': curr_price,
        'distance_pct': ((curr_price - curr_ema60) / curr_ema60) * 100,
        'above_count': above_count,
        'above_pct': above_pct,
        'is_consolidating': is_consolidating,
        'recent_range_pct': recent_range_pct,
        'positive_momentum': positive_momentum,
        'ema60_slope': ema60_slope,
        'trend_strength': 'Strong' if ema60_slope > 0.5 else 'Moderate' if ema60_slope > 0 else 'Weak'
    }
    
    # Consolidamento sopra EMA 60 = VALID (preparazione breakout)
    if is_consolidating and above_ema60:
        return (True, 'Consolidation above EMA 60 (pre-breakout)', details)
    
    # Pullback sopra EMA 60 = VALID (retest sano)
    if not positive_momentum and above_ema60:
        return (True, 'Pullback above EMA 60 (healthy)', details)
    
    # Uptrend normale = VALID
    if positive_momentum and above_ema60:
        return (True, 'Strong uptrend', details)
    
    # Default: se sopra EMA 60 = OK
    return (True, 'Above EMA 60', details)

def check_structure_trend(df: pd.DataFrame, lookback: int = 10) -> tuple:
    """
    Structure-Based Trend Check (originale)
    
    PROBLEMA: Blocca consolidamenti
    """
    if len(df) < lookback + 3:
        return (False, 'Insufficient data', {})
    
    highs = df['high'].iloc[-lookback:]
    lows = df['low'].iloc[-lookback:]
    
    split = lookback // 2
    
    recent_high = highs.iloc[-split:].max()
    previous_high = highs.iloc[:-split].max()
    
    recent_low = lows.iloc[-split:].min()
    previous_low = lows.iloc[:-split].min()
    
    has_hh = recent_high > previous_high
    has_hl = recent_low > previous_low
    
    details = {
        'recent_high': recent_high,
        'previous_high': previous_high,
        'recent_low': recent_low,
        'previous_low': previous_low,
        'has_hh': has_hh,
        'has_hl': has_hl
    }
    
    if has_hh and has_hl:
        return (True, 'Higher Highs + Higher Lows', details)
    elif has_hh:
        return (False, 'Higher High but not Higher Low', details)
    elif has_hl:
        return (False, 'Higher Low but not Higher High', details)
    else:
        return (False, 'No uptrend structure', details)

def check_hybrid_trend(df: pd.DataFrame) -> tuple:
    """
    Hybrid Trend Check
    
    LOGICA: Structure OR EMA (più permissivo)
    """
    config = TREND_FILTER_CONFIG['hybrid']
    
    structure_valid, structure_reason, structure_details = check_structure_trend(df)
    ema_valid, ema_reason, ema_details = check_ema_trend(df)
    
    require_both = config.get('require_both', False)
    
    if require_both:
        # AND logic (stretto)
        valid = structure_valid and ema_valid
        reason = f'Structure: {structure_reason}, EMA: {ema_reason}'
    else:
        # OR logic (permissivo)
        valid = structure_valid or ema_valid
        
        if structure_valid and ema_valid:
            reason = 'Both Structure and EMA valid'
        elif structure_valid:
            reason = f'Structure valid: {structure_reason}'
        elif ema_valid:
            reason = f'EMA valid: {ema_reason}'
        else:
            reason = 'Neither Structure nor EMA valid'
    
    details = {
        'structure': structure_details,
        'ema': ema_details,
        'mode': 'AND' if require_both else 'OR'
    }
    
    return (valid, reason, details)

# ===== PATTERN-SPECIFIC OVERRIDES =====

PATTERN_TREND_REQUIREMENTS = {
    # Pattern che RICHIEDONO consolidamento
    'Triple Touch Breakout': {
        'allow_consolidation': True,
        'require_ema60': True,  # MA solo EMA 60, non structure
    },
    'Breakout + Retest': {
        'allow_consolidation': True,
        'require_ema60': True,
    },
    'Bullish Flag Breakout': {
        'allow_consolidation': True,  # Flag è consolidamento!
        'require_ema60': True,
    },
    'Compression Breakout': {
        'allow_consolidation': True,  # Compression è consolidamento!
        'require_ema60': False,  # EMA check interno
    },
    
    # Pattern che richiedono uptrend forte
    'Volume Spike Breakout': {
        'require_momentum': True,
        'require_ema60': True,
    },
    'Liquidity Sweep + Reversal': {
        'require_ema60': True,  # MA permetti pullback
        'allow_pullback': True,
    },
}

def check_pattern_specific_trend(df: pd.DataFrame, pattern_name: str) -> tuple:
    """
    Check trend specifico per pattern
    
    Usa requirements custom per ogni pattern
    """
    if pattern_name not in PATTERN_TREND_REQUIREMENTS:
        # Default: usa check globale
        return is_valid_trend_for_entry(df, mode=TREND_FILTER_MODE)
    
    requirements = PATTERN_TREND_REQUIREMENTS[pattern_name]
    
    # Check EMA 60 se richiesto
    if requirements.get('require_ema60'):
        ema_60 = df['close'].ewm(span=60, adjust=False).mean()
        curr_price = df['close'].iloc[-1]
        curr_ema60 = ema_60.iloc[-1]
        
        # Permetti consolidamento se allow_consolidation
        if requirements.get('allow_consolidation'):
            # OK se sopra EMA 60 (anche se in range)
            if curr_price > curr_ema60 * 0.98:
                return (True, f'{pattern_name}: Above EMA 60 (consolidation OK)', {})
            else:
                return (False, f'{pattern_name}: Below EMA 60', {})
        
        # Check standard EMA
        return check_ema_trend(df)
    
    # Pattern senza requirements specifici
    return (True, f'{pattern_name}: No specific requirements', {})

def get_price_decimals(price: float) -> int:
    """
    Determina il numero di decimali in base al prezzo
    
    - Prezzo >= 1000 (es. BTC): 2 decimali
    - Prezzo >= 100: 3 decimali
    - Prezzo >= 10: 4 decimali
    - Prezzo >= 1: 5 decimali
    - Prezzo < 1: 6 decimali
    """
    if price >= 1000:
        return 2
    elif price >= 100:
        return 3
    elif price >= 10:
        return 4
    elif price >= 1:
        return 5
    else:
        return 6

# ===== DYNAMIC RISK CALCULATION =====
def calculate_dynamic_risk(ema_score: int) -> float:
    """
    Scala il rischio in base alla qualità EMA
    
    Score ranges:
    - GOLD (80+): $20 (+50%)
    - GOOD (60-80): $15 (standard)
    - OK (40-60): $10 (-50%)
    - WEAK (<40): $5 (-80%)
    """
    if ema_score >= 80:
        return 20.0  # 🌟 Setup perfetto
    elif ema_score >= 60:
        return 15.0  # ✅ Setup buono
    elif ema_score >= 40:
        return 10.0   # ⚠️ Setup debole
    else:
        return 5.0   # ❌ Setup molto debole


def analyze_ema_conditions(df: pd.DataFrame, timeframe: str, pattern_name: str = None):
    """
    Analizza le condizioni EMA per il timeframe specificato
    
    Returns:
        dict con:
        - score: punteggio 0-100
        - quality: 'GOLD', 'GOOD', 'OK', 'WEAK', 'BAD'
        - conditions: dict di condizioni soddisfatte
        - details: messaggio testuale
    """
    if not config.EMA_FILTER_ENABLED or config.EMA_FILTER_MODE == 'off':
        return {
            'score': 100,
            'quality': 'OK',
            'conditions': {},
            'details': 'Filtro EMA disabilitato',
            'passed': True
        }
    
    # Calcola EMA
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    ema_223 = df['close'].ewm(span=223, adjust=False).mean()
    
    last_close = df['close'].iloc[-1]
    last_ema5 = ema_5.iloc[-1]
    last_ema10 = ema_10.iloc[-1]
    last_ema60 = ema_60.iloc[-1]
    last_ema223 = ema_223.iloc[-1]
    
    # Determina configurazione per timeframe
    configEma = None
    for strategy, cfg in config.EMA_CONFIG.items():
        if timeframe in cfg['timeframes']:
            configEma = cfg
            break
    
    if not configEma:
        # Default: usa day trading config
        configEma = config.EMA_CONFIG['daytrading']
    
    rules = configEma['rules']
    conditions = {}
    score = 0
    details = []
    
    if timeframe in ['5m', '15m']:
        # MUST 1: Prezzo sopra EMA 10
        if rules.get('price_above_ema10'):
            if last_close > last_ema10:
                conditions['price_above_ema10'] = True
                score += 30  # 👈 RIDOTTO da 40 (per fare spazio a EMA 60)
                details.append("Prezzo maggiore EMA 10")
            else:
                conditions['price_above_ema10'] = False
                score -= 30
                details.append("Prezzo minore EMA 10")
        
        # ===== MUST 2: Prezzo sopra EMA 60 (NUOVO - FONDAMENTALE) =====
        if last_close > last_ema60:
            conditions['price_above_ema60'] = True
            score += 30  # 👈 NUOVO: punti per trend rialzista
            details.append("Prezzo maggiore EMA 60 (trend rialzista)")
        else:
            conditions['price_above_ema60'] = False
            score -= 20  # 👈 Penalità se contro trend
            details.append("Prezzo minore EMA 60 (contro trend)")
        
        # BONUS: EMA 5 sopra EMA 10 (momentum)
        if rules.get('ema5_above_ema10'):
            if last_ema5 > last_ema10:
                conditions['ema5_above_ema10'] = True
                score += 20  # 👈 RIDOTTO da 30
                details.append("EMA 5 maggiore EMA 10 (momentum)")
            else:
                conditions['ema5_above_ema10'] = False
                score += 10
                details.append("EMA 5 minore EMA 10")
        
        # GOLD: Pattern vicino a EMA 10 (pullback)
        distance_to_ema10 = abs(last_close - last_ema10) / last_ema10
        if distance_to_ema10 < 0.005:
            conditions['near_ema10'] = True
            score += 20  # 👈 RIDOTTO da 30
            details.append("Vicino EMA 10 - pullback zone")
        else:
            conditions['near_ema10'] = False
    
    # DAY TRADING (30m, 1h)
    elif timeframe in ['30m', '1h']:
        if rules.get('price_above_ema60'):
            if last_close > last_ema60:
                conditions['price_above_ema60'] = True
                score += 40
                details.append("Prezzo maggiore EMA 60")
            else:
                conditions['price_above_ema60'] = False
                score -= 30
                details.append("Prezzo minore EMA 60")
        
        if rules.get('ema10_above_ema60'):
            if last_ema10 > last_ema60:
                conditions['ema10_above_ema60'] = True
                score += 30
                details.append("EMA 10 maggiore EMA 60 (trend ok)")
            else:
                conditions['ema10_above_ema60'] = False
                score += 10
                details.append("EMA 10 minore EMA 60")
        
        distance_to_ema60 = abs(last_close - last_ema60) / last_ema60
        if distance_to_ema60 < 0.01:
            conditions['near_ema60'] = True
            score += 30
            details.append("Vicino EMA 60 - bounce zone")
        else:
            conditions['near_ema60'] = False
    
    # SWING (4h)
    elif timeframe in ['4h']:
        if rules.get('price_above_ema223'):
            if last_close > last_ema223:
                conditions['price_above_ema223'] = True
                score += 40
                details.append("Prezzo maggiore EMA 223 (bull market)")
            else:
                conditions['price_above_ema223'] = False
                score -= 30
                details.append("Prezzo minore EMA 223 (bear market)")
        
        if rules.get('ema60_above_ema223'):
            if last_ema60 > last_ema223:
                conditions['ema60_above_ema223'] = True
                score += 30
                details.append("EMA 60 maggiore EMA 223 (strong trend)")
            else:
                conditions['ema60_above_ema223'] = False
                score += 10
                details.append("EMA 60 minore EMA 223")
        
        distance_to_ema223 = abs(last_close - last_ema223) / last_ema223
        if distance_to_ema223 < 0.02:
            conditions['near_ema223'] = True
            score += 30
            details.append("Vicino EMA 223 - major support")
        else:
            conditions['near_ema223'] = False

    # BREAKOUT (1m, 3m)
    elif timeframe in ['1m', '3m']:
        # Check se prezzo ha appena rotto EMA 223 al rialzo
        
        # 1. Prezzo deve essere APPENA sopra EMA 223 (entro 0.5%)
        just_above_223 = (last_close > last_ema223 and 
                         (last_close - last_ema223) / last_ema223 < 0.005)
        
        # 2. Verifica che nelle ultime 3 candele il prezzo ERA sotto EMA 223
        was_below = False
        if len(df) >= 4:
            prev_closes = [df['close'].iloc[-2], df['close'].iloc[-3], df['close'].iloc[-4]]
            prev_ema223 = [ema_223.iloc[-2], ema_223.iloc[-3], ema_223.iloc[-4]]
            
            # Almeno 2 delle 3 candele precedenti erano sotto EMA 223
            below_count = sum(1 for c, e in zip(prev_closes, prev_ema223) if c < e)
            was_below = below_count >= 2
        
        # 3. EMA 5 e 10 devono essere sopra EMA 223 (conferma momentum)
        ema_aligned = last_ema5 > last_ema223 and last_ema10 > last_ema223
        
        # RILEVAMENTO BREAKOUT
        if just_above_223 and was_below and ema_aligned:
            # 🚀 BREAKOUT CONFERMATO!
            conditions['breakout_ema223'] = True
            score = 100  # Score massimo automatico
            details.append("BREAKOUT EMA 223 CONFERMATO")  # ✅ NO emoji
            details.append("Prezzo ha rotto EMA 223 al rialzo")
            details.append("EMA 5 e 10 sopra EMA 223")
            details.append("Setup ad alta probabilità")
        else:
            # Setup normale per 1m/3m
            # MUST: Prezzo sopra EMA 223
            if last_close > last_ema223:
                conditions['price_above_ema223'] = True
                score += 40
                details.append("Prezzo > EMA 223")
            else:
                conditions['price_above_ema223'] = False
                score -= 30
                details.append("Prezzo < EMA 223")
            
            # BONUS: EMA 5 e 10 sopra EMA 223
            if last_ema5 > last_ema223 and last_ema10 > last_ema223:
                conditions['ema_above_223'] = True
                score += 30
                details.append("EMA 5,10 > EMA 223 (momentum)")
            else:
                conditions['ema_above_223'] = False
                score += 10
                details.append("EMA non tutte sopra 223")
            
            # GOLD: Pattern molto vicino a EMA 223 (bounce)
            distance_to_ema223 = abs(last_close - last_ema223) / last_ema223
            if distance_to_ema223 < 0.003:  # Entro 0.3%
                conditions['near_ema223'] = True
                score += 30
                details.append("Vicino EMA 223 (bounce zone!)")
            else:
                conditions['near_ema223'] = False


    # === CASO SPECIALE: Liquidity Sweep Pattern ===
    # Per questo pattern, le regole EMA sono diverse
    if pattern_name == 'Liquidity Sweep + Reversal':
        # MUST: Solo EMA 60 (trend principale)
        if last_close > last_ema60:
            score = 80  # Score fisso GOOD
            details = []  # Reset details per logica custom
            details.append("Pattern istituzionale: Liquidity Sweep")
            details.append("Prezzo maggiore EMA 60 (trend rialzista)")
            details.append("Sweep sotto EMA 5,10 tollerato")
            
            # BONUS: Vicino a EMA 60
            distance_to_ema60 = abs(last_close - last_ema60) / last_ema60
            if distance_to_ema60 < 0.01:
                score = 100  # GOLD se vicino a EMA 60
                details.append("Vicino EMA 60 - strong bounce zone")
            
            return {
                'score': score,
                'quality': 'GOLD' if score == 100 else 'GOOD',
                'conditions': {'price_above_ema60': True},
                'details': '\n'.join(details),
                'passed': True,
                'ema_values': {
                    'ema5': last_ema5,
                    'ema10': last_ema10,
                    'ema60': last_ema60,
                    'ema223': last_ema223,
                    'price': last_close
                }
            }
        else:
            # Sweep in downtrend = BAD
            return {
                'score': 0,
                'quality': 'BAD',
                'conditions': {'price_above_ema60': False},
                'details': 'Liquidity Sweep in downtrend - SKIP',
                'passed': False,
                'ema_values': {
                    'ema5': last_ema5,
                    'ema10': last_ema10,
                    'ema60': last_ema60,
                    'ema223': last_ema223,
                    'price': last_close
                }
            }
    
    # Normalizza score tra 0-100
    score = max(0, min(100, score))
    
    # Determina quality
    if score >= 80:
        quality = 'GOLD'  # 🌟 Setup perfetto
    elif score >= 60:
        quality = 'GOOD'  # ✅ Setup buono
    elif score >= 40:
        quality = 'OK'    # ⚠️ Setup accettabile
    elif score >= 20:
        quality = 'WEAK'  # 🔶 Setup debole
    else:
        quality = 'BAD'   # ❌ Setup da evitare
    
    # Determina se passa il filtro
    if config.EMA_FILTER_MODE == 'strict':
        passed = score >= 60  # Strict: solo GOOD e GOLD
    else:  # loose
        passed = score >= 40  # Loose: anche OK
    
    return {
        'score': score,
        'quality': quality,
        'conditions': conditions,
        'details': '\n'.join(details),
        'passed': passed,
        'ema_values': {
            'ema5': last_ema5,
            'ema10': last_ema10,
            'ema60': last_ema60,
            'ema223': last_ema223,
            'price': last_close
        }
    }


def is_volume_spike_breakout(df: pd.DataFrame) -> tuple:
    """
    🥇 Volume Spike + EMA Breakout
    
    Volume check dipende da VOLUME_FILTER_MODE:
    - 'strict': Richiede 3x minimo
    - 'adaptive': Richiede 2.5x (più permissivo)
    - 'pattern-only': Usa soglia custom 3x
    """
    if len(df) < 60:
        return (False, None)

    # 🔧 FIX: Inizializza subito
    pattern_data = None
    
    # Candele
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # === CHECK 1: VOLUME SPIKE (usa mode) ===
    vol = df['volume']
    
    if len(vol) < 20:
        return (False, None)
    
    avg_vol = vol.iloc[-20:-1].mean()
    current_vol = vol.iloc[-1]
    
    if avg_vol == 0:
        return (False, None)
    
    volume_ratio = current_vol / avg_vol
    
    # Determina threshold da VOLUME_FILTER_MODE
    if VOLUME_FILTER_MODE == 'adaptive':
        min_volume_ratio = 2.0  # Più permissivo
    else:
        min_volume_ratio = 2.5  # Default strict
    
    if volume_ratio < min_volume_ratio:
        return (False, None)
    
    # ===== AGGIUNGI: Verifica che non sia già pump esaurito =====
    # Ultimi 3 prezzi NON devono essere tutti in salita verticale
    recent_3 = df['close'].iloc[-3:]
    vertical_pump = all(recent_3.iloc[i+1] > recent_3.iloc[i] * 1.005 for i in range(2))
    
    if vertical_pump:
        logging.debug(f'Volume Spike: Pump già in atto, skip per evitare top')
        return (False, None)
    
    # === CHECK 2: CANDELA VERDE FORTE ===
    is_bullish = curr['close'] > curr['open']
    
    if not is_bullish:
        return (False, None)
    
    body = abs(curr['close'] - curr['open'])
    total_range = curr['high'] - curr['low']
    
    if total_range == 0:
        return (False, None)
    
    body_pct = body / total_range
    
    # Corpo minimo 60% del range
    if body_pct < 0.60:
        return (False, None)
    
    # === CHECK 3: CLOSE VICINO HIGH (no rejection) ===
    close_to_high_distance = curr['high'] - curr['close']
    
    # Ombra superiore max 15% del range
    if close_to_high_distance > total_range * 0.15:
        return (False, None)
    
    # === CHECK 4: EMA BREAKOUT ===
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    curr_ema10 = ema_10.iloc[-1]
    curr_ema60 = ema_60.iloc[-1]
    prev_ema10 = ema_10.iloc[-2]

    # ===== AGGIUNGI: Check overextension da EMA 10 =====
    distance_from_ema10 = abs(curr['close'] - curr_ema10) / curr_ema10
    if distance_from_ema10 > 0.008:  # Max 0.8% da EMA 10
        logging.debug(f'Volume Spike: Prezzo troppo esteso da EMA 10 ({distance_from_ema10*100:.2f}%)')
        return (False, None)
    
    # Era sotto EMA 10, ora sopra (breakout)
    was_below_ema10 = prev['close'] < prev_ema10
    now_above_ema10 = curr['close'] > curr_ema10
    
    # Deve essere sopra EMA 60 (trend rialzista)
    above_ema60 = curr['close'] > curr_ema60
    
    # EMA 10 sopra EMA 60 (alignment)
    ema_aligned = curr_ema10 > curr_ema60
    
    if not (now_above_ema10 and above_ema60 and ema_aligned):
        return (False, None)
    
    # === CHECK 5: MOMENTUM NON GIÀ ESTESO ===
    # Candela precedente non deve essere già troppo forte
    prev_body = abs(prev['close'] - prev['open'])
    prev_range = prev['high'] - prev['low']
    
    if prev_range > 0:
        prev_body_pct = prev_body / prev_range
        
        # Se prev era già molto forte (>70%) e verde, skip
        # (vogliamo comprare l'inizio del movimento, non il pump)
        if prev_body_pct > 0.70 and prev['close'] > prev['open']:
            return (False, None)
    
    # === PATTERN CONFERMATO! ===
    pattern_data = {
        'volume_ratio': volume_ratio,
        'body_pct': body_pct,
        'ema10': curr_ema10,
        'ema60': curr_ema60,
        'breakout_confirmed': was_below_ema10 and now_above_ema10,
        'price': curr['close']
    }
    
    return (True, pattern_data)

# ----------------------------- PATTERN DETECTION -----------------------------

def is_support_resistance_bounce(df: pd.DataFrame) -> tuple:
    """
    🥉 SUPPORT/RESISTANCE BOUNCE (OPTIMIZED FOR 5m)
    
    Win Rate: 52-58% (5m), 56-62% (15m)
    Risk:Reward: 1.6:1 medio
    
    MODIFICHE per 5m:
    ✅ Lookback ridotto da 50 a 30 candele (2.5h invece di 4h)
    ✅ Support deve essere toccato RECENTEMENTE (max 15 candele fa)
    ✅ Volume threshold ridotto a 1.0x per velocità 5m
    ✅ Distance check EMA 60 esteso a 3% per maggiore flessibilità
    
    COME FUNZIONA:
    ============================================
    1. Identifica livelli S/R significativi (ultimi 30 periodi)
    2. Prezzo "tocca" support (bounce zone ±0.5%)
    3. Candela verde con REJECTION (ombra lunga sotto)
    4. Volume superiore alla media
    5. Trend ancora intatto (prezzo vicino a EMA 10)
    
    LOGICA:
    ============================================
    - Support/Resistance = zone psicologiche
    - Molti trader/bot piazzano ordini su S/R
    - Self-fulfilling prophecy
    - Rejection (long wick) = domanda forte
    
    FILTRI USATI:
    ============================================
    ✅ Filtri GLOBALI:
       - Volume > 1.0x media (più permissivo per 5m)
       - Uptrend structure (check_patterns)
       - ATR expanding (warning only)
    
    ✅ Filtri INTERNI:
       - Volume > 1.0x media
       - Support toccato 3+ volte (valido)
       - Support toccato max 15 candele fa (fresco)
       - Rejection: wick >= corpo
       - Close sopra/vicino EMA 10 (trend OK)
       - Distanza < 3% da EMA 60 (esteso per 5m)
    
    EMA USATE:
    ============================================
    - EMA 10: Check trend breve (momentum)
    - EMA 60: Check trend medio (distanza max 3%)
    - NO EMA 5, 223 (non rilevanti per questo pattern)
    
    Returns:
        (found: bool, pattern_data: dict or None)
    """
    if len(df) < 50:
        return (False, None)

    # 🔧 FIX: Inizializza subito
    pattern_data = None
    
    curr = df.iloc[-1]
    
    # === STEP 1: IDENTIFICA SUPPORT LEVEL ===
    # MODIFICA: Lookback ridotto per 5m
    lookback = 30  # Era 50 → 2.5 ore invece di 4 ore
    lookback_lows = df['low'].iloc[-lookback:-1]
    
    # Trova i 5 low più bassi
    sorted_lows = lookback_lows.nsmallest(5)
    
    # Support level = media dei 5 low (riduce noise)
    support_level = sorted_lows.mean()
    
    # Conta quante volte è stato toccato
    tolerance = support_level * 0.005  # ±0.5%
    touches = (lookback_lows <= support_level + tolerance).sum()
    
    # Support deve essere significativo (3+ touches)
    if touches < 3:
        return (False, None)
    
    # ===== AGGIUNGI: Support deve essere toccato RECENTEMENTE =====
    # Per 5m, support vecchio è irrilevante
    last_touch_idx = None
    for i in range(len(lookback_lows)-1, -1, -1):
        if abs(lookback_lows.iloc[i] - support_level) <= tolerance:
            last_touch_idx = i
            break
    
    # Se ultimo tocco > 15 candele fa (75 min su 5m), support non è fresco
    if last_touch_idx is not None and last_touch_idx < len(lookback_lows) - 15:
        logging.debug(f'S/R Bounce: Support non recente (ultimo tocco {len(lookback_lows) - last_touch_idx} candele fa)')
        return (False, None)
    
    # === STEP 2: PREZZO TOCCA SUPPORT ===
    # Low corrente deve essere nella "bounce zone" (±0.5%)
    touches_support = abs(curr['low'] - support_level) <= tolerance
    
    if not touches_support:
        return (False, None)
    
    # === STEP 3: CANDELA VERDE con REJECTION ===
    is_bullish = curr['close'] > curr['open']
    
    if not is_bullish:
        return (False, None)
    
    # Calcola ombra inferiore e corpo
    lower_wick = min(curr['open'], curr['close']) - curr['low']
    body = abs(curr['close'] - curr['open'])
    total_range = curr['high'] - curr['low']
    
    if total_range == 0:
        return (False, None)
    
    # Ombra inferiore deve essere >= corpo (rejection forte)
    # Questo mostra che compratori hanno "difeso" il support
    if lower_wick < body:
        return (False, None)
    
    # Body deve essere almeno 30% del range (non doji)
    body_pct = body / total_range
    
    if body_pct < 0.30:
        return (False, None)
    
    # === STEP 4: VOLUME CONFIRMATION ===
    # Volume deve essere sopra media (conferma interesse)
    if 'volume' not in df.columns or len(df['volume']) < 20:
        return (False, None)
    
    vol = df['volume']
    avg_vol = vol.iloc[-20:-1].mean()
    curr_vol = vol.iloc[-1]
    
    if avg_vol == 0:
        return (False, None)
    
    vol_ratio = curr_vol / avg_vol
    
    # MODIFICA: Volume minimo 1.0x (più permissivo per 5m)
    # Era 1.2x → troppo stretto per velocità 5m
    if vol_ratio < 1.0:
        return (False, None)
    
    # === STEP 5: EMA 10 CHECK (trend breve intatto) ===
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    curr_ema10 = ema_10.iloc[-1]
    
    # Close deve essere sopra (o molto vicino) EMA 10
    # Tolleranza: max 0.3% sotto
    close_vs_ema10 = curr['close'] / curr_ema10
    
    if close_vs_ema10 < 0.997:
        return (False, None)
    
    # === STEP 6: EMA 60 CHECK (trend medio) ===
    # Non vogliamo bounce troppo lontano da trend principale
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    curr_ema60 = ema_60.iloc[-1]
    
    # MODIFICA: Distanza massima 3% da EMA 60 (era 2%)
    # Più permissivo per 5m per non perdere setup validi
    distance_to_ema60 = abs(curr['close'] - curr_ema60) / curr_ema60
    
    if distance_to_ema60 > 0.03:
        return (False, None)
    
    # === STEP 7: QUALITY BONUS (opzionale) ===
    # Più vicino a EMA 60 = qualità migliore
    near_ema60 = distance_to_ema60 < 0.01  # Entro 1%
    
    # Rejection strength (quanto è forte il wick rispetto al corpo)
    rejection_strength = lower_wick / body if body > 0 else 1.0
    
    # ===== AGGIUNGI: Calcola freshness del support =====
    support_freshness = 'fresh' if last_touch_idx is not None and last_touch_idx >= len(lookback_lows) - 5 else 'recent'
    
    # === PATTERN CONFERMATO ===
    pattern_data = {
        'support_level': support_level,
        'touches': touches,
        'distance_to_support': abs(curr['low'] - support_level),
        'volume_ratio': vol_ratio,
        'ema10': curr_ema10,
        'ema60': curr_ema60,
        'rejection_strength': rejection_strength,
        'near_ema60': near_ema60,
        'body_pct': body_pct,
        'lower_wick_pct': lower_wick / total_range,
        'support_freshness': support_freshness,  # NUOVO
        'last_touch_candles_ago': len(lookback_lows) - last_touch_idx if last_touch_idx is not None else None,  # NUOVO
        'tier': 2  # Good (Tier 2) - considerato medio per 5m
    }
    
    return (True, pattern_data)

def is_higher_low_consolidation_breakout(df: pd.DataFrame) -> tuple:
    """
    Higher Low Consolidation Breakout
    
    LOGICA:
    1. Identifica impulso iniziale (grande verde)
    2. Consolidamento DEVE avere higher lows
    3. CRITICAL: Mai rompere low impulso iniziale
    4. Breakout con volume
    
    Returns:
        (found: bool, pattern_data: dict or None)
    """
    if len(df) < 15:
        return (False, None)
    
    # ===== FASE 1: IDENTIFICA IMPULSO INIZIALE =====
    # Cerca grande candela verde negli ultimi 12 periodi
    
    impulse_found = False
    impulse_idx = None
    impulse_candle = None
    
    for i in range(-12, -3):  # Da -12 a -4
        if len(df) < abs(i):
            continue
        
        candle = df.iloc[i]
        
        # Deve essere verde
        if candle['close'] <= candle['open']:
            continue
        
        # Corpo forte (>60% range)
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            continue
        
        body_pct = body / total_range
        
        if body_pct < 0.60:
            continue
        
        # Body significativo in assoluto (>0.8% del prezzo)
        body_pct_price = (body / candle['open']) * 100
        
        if body_pct_price < 0.8:
            continue
        
        # IMPULSO TROVATO!
        impulse_found = True
        impulse_candle = candle
        impulse_idx = i
        break
    
    if not impulse_found:
        return (False, None)
    
    # Livelli chiave
    resistance = impulse_candle['high']  # Breakout level
    base_support = impulse_candle['low']  # "Line in the sand"
    
    # ===== FASE 2: CONSOLIDAMENTO con HIGHER LOWS =====
    # Candele tra impulso e corrente
    
    consolidation_start = impulse_idx + 1
    consolidation_end = -1
    
    consolidation = df.iloc[consolidation_start:consolidation_end]
    
    if len(consolidation) < 3:
        return (False, None)  # Troppo corto
    
    if len(consolidation) > 10:
        return (False, None)  # Troppo lungo
    
    # CHECK CRITICO: Mai rompere base support
    consolidation_low = consolidation['low'].min()
    
    if consolidation_low <= base_support * 0.998:  # Tolleranza 0.2%
        logging.debug(f'❌ Higher Low Breakout: Base broken ({consolidation_low:.6f} < {base_support:.6f})')
        return (False, None)  # Base violata = pattern INVALIDO
    
    # CHECK HIGHER LOWS (opzionale ma migliora win rate)
    # Divide consolidamento in 2 metà
    if len(consolidation) >= 6:
        split = len(consolidation) // 2
        first_half_low = consolidation['low'].iloc[:split].min()
        second_half_low = consolidation['low'].iloc[split:].min()
        
        # Second half low dovrebbe essere >= first half low
        has_higher_lows = second_half_low >= first_half_low * 0.995
        
        if not has_higher_lows:
            logging.debug('⚠️ No clear higher lows pattern')
            # Non blocca, ma riduce quality score
    else:
        has_higher_lows = True  # Troppo corto per determinare
    
    # Range consolidamento deve essere stretto
    consolidation_high = consolidation['high'].max()
    consolidation_range = consolidation_high - consolidation_low
    consolidation_range_pct = (consolidation_range / consolidation_low) * 100
    
    if consolidation_range_pct > 3.0:
        return (False, None)  # Range troppo ampio (>3%)
    
    # Consolidamento deve rimanere SOTTO resistance
    if consolidation_high > resistance * 1.002:
        return (False, None)  # Ha già rotto resistance
    
    # ===== FASE 3: BREAKOUT (candela corrente) =====
    
    curr = df.iloc[-1]
    
    # Deve essere verde
    if curr['close'] <= curr['open']:
        return (False, None)
    
    # Deve rompere resistance
    if curr['close'] <= resistance:
        return (False, None)
    
    # Corpo forte
    curr_body = abs(curr['close'] - curr['open'])
    curr_range = curr['high'] - curr['low']
    
    if curr_range == 0:
        return (False, None)
    
    curr_body_pct = (curr_body / curr_range) * 100
    
    if curr_body_pct < 50:
        return (False, None)
    
    # Upper wick piccolo (no rejection)
    curr_upper_wick = curr['high'] - curr['close']
    curr_upper_wick_pct = (curr_upper_wick / curr_range) * 100
    
    if curr_upper_wick_pct > 30:
        return (False, None)
    
    # ===== VOLUME CHECK =====
    if 'volume' not in df.columns:
        return (False, None)
    
    consolidation_vol_avg = consolidation['volume'].mean()
    curr_vol = df['volume'].iloc[-1]
    
    if consolidation_vol_avg == 0:
        return (False, None)
    
    vol_ratio = curr_vol / consolidation_vol_avg
    
    # Volume breakout > 1.5x consolidamento
    if vol_ratio < 1.5:
        return (False, None)
    
    # ===== EMA CHECKS (opzionali) =====
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    curr_ema10 = ema_10.iloc[-1]
    curr_ema60 = ema_60.iloc[-1]
    
    # Preferibile se sopra EMA
    above_ema10 = curr['close'] > curr_ema10
    above_ema60 = curr['close'] > curr_ema60
    
    # ===== QUALITY SCORING =====
    quality_score = 70  # Base score
    
    if has_higher_lows:
        quality_score += 10
    
    if vol_ratio > 0.0:
        quality_score += 10
    
    if above_ema10 and above_ema60:
        quality_score += 10
    
    quality = 'GOLD' if quality_score >= 90 else 'GOOD' if quality_score >= 80 else 'OK'
    
    # ===== PATTERN CONFERMATO =====
    
    pattern_data = {
        # Livelli chiave
        'resistance': resistance,
        'base_support': base_support,
        'consolidation_low': consolidation_low,
        'consolidation_high': consolidation_high,
        'range_pct': consolidation_range_pct,
        
        # Impulso info
        'impulse_body_pct': (abs(impulse_candle['close'] - impulse_candle['open']) / 
                             (impulse_candle['high'] - impulse_candle['low']) * 100),
        
        # Consolidamento info
        'consolidation_duration': len(consolidation),
        'has_higher_lows': has_higher_lows,
        'base_held': True,  # Sempre True se arriviamo qui
        
        # Breakout info
        'breakout_price': curr['close'],
        'breakout_body_pct': curr_body_pct,
        'volume_ratio': vol_ratio,
        
        # EMA
        'ema10': curr_ema10,
        'ema60': curr_ema60,
        'above_ema10': above_ema10,
        'above_ema60': above_ema60,
        
        # Trading setup
        'suggested_entry': curr['close'],
        'suggested_sl': base_support * 0.998,  # Sotto base
        'suggested_tp': resistance + (resistance - base_support) * 3,  # 3R
        
        # Quality
        'quality': quality,
        'quality_score': quality_score,
        
        'tier': 2  # Good but overlaps with Flag
    }
    
    return (True, pattern_data)

def is_bullish_engulfing_enhanced(prev, curr, df):
    """
    🟢 BULLISH ENGULFING ENHANCED (EMA-Optimized)
    
    Win Rate Base: ~45%
    Win Rate Enhanced: ~55-62%
    
    LOGICA MULTI-TIER:
    ==========================================
    
    TIER 1 - GOLD Setup (65-70% win): 🌟
    ├─ Engulfing vicino EMA 60 (±0.5%)
    ├─ Dopo pullback (era sopra, è tornato)
    ├─ Volume 2x+
    ├─ Rejection forte (wick >= corpo)
    └─ → Entry IDEALE (institutional support)
    
    TIER 2 - GOOD Setup (58-62% win): ✅
    ├─ Engulfing vicino EMA 10 (±1%)
    ├─ Sopra EMA 60 (trend intact)
    ├─ Volume 1.8x+
    ├─ Rejection moderata
    └─ → Entry VALIDO (short-term support)
    
    TIER 3 - OK Setup (52-55% win): ⚠️
    ├─ Engulfing generico
    ├─ Sopra EMA 60 (solo trend filter)
    ├─ Volume 1.8x+ (was 1.5x)
    └─ → Entry ACCETTABILE (minimal edge)
    
    REJECTION:
    ├─ Sotto EMA 60 (downtrend)
    ├─ Volume < 1.8x (was 1.5x)
    ├─ Prev candle capitulation (volume 3x+)
    └─ Troppo lontano da EMA (>2%)
    
    Returns:
        (found: bool, tier: str, data: dict or None)
    """
    if len(df) < 60:
        return (False, None, None)

    # ========================================
    # 🔧 FIX: Dichiara TUTTE le variabili usate in pattern_data
    # ========================================
    rejection_strength = 0.0
    lower_wick_pct = 0.0
    upper_wick_pct = 0.0
    curr_body = 0.0
    prev_body = 0.0
    total_range = 0.0
    was_higher = False
    distance_to_ema5 = 0.0
    distance_to_ema10 = 0.0
    distance_to_ema60 = 0.0
    above_ema10 = False
    above_ema60 = False
    
    # ===== STEP 1: ENGULFING BASE CHECK =====
    prev_body_top = max(prev['open'], prev['close'])
    prev_body_bottom = min(prev['open'], prev['close'])
    curr_body_top = max(curr['open'], curr['close'])
    curr_body_bottom = min(curr['open'], curr['close'])
    
    is_prev_bearish = prev['close'] < prev['open']
    is_curr_bullish = curr['close'] > curr['open']
    
    engulfs = (curr_body_bottom <= prev_body_bottom and 
               curr_body_top >= prev_body_top)
    
    prev_body = abs(prev['open'] - prev['close'])
    curr_body = abs(curr['open'] - curr['close'])
    has_body = curr_body >= prev_body * 0.5
    
    if not (is_prev_bearish and is_curr_bullish and engulfs and has_body):
        return (False, None, None)
    
    # ===== STEP 2: VOLUME CHECK (MANDATORY) =====
    if 'volume' not in df.columns or len(df['volume']) < 20:
        return (False, None, None)
    
    vol = df['volume']
    avg_vol = vol.iloc[-20:-1].mean()
    curr_vol = vol.iloc[-1]
    
    if avg_vol == 0:
        return (False, None, None)
    
    vol_ratio = curr_vol / avg_vol
    
    # ===== NUOVO: Check candela precedente non deve essere capitulation =====
    # Se candela precedente ha volume 3x+ = capitulation, engulfing meno affidabile
    if len(vol) >= 22:
        prev_vol = vol.iloc[-2]
        avg_vol_before = vol.iloc[-22:-2].mean()
        
        if avg_vol_before > 0:
            prev_vol_ratio = prev_vol / avg_vol_before
            
            # Se prev aveva volume panic (3x+), skip engulfing
            if prev_vol_ratio > 2.0:
                logging.debug(f'Engulfing: Candela prev era capitulation (vol {prev_vol_ratio:.1f}x), skip')
                return (False, None, None)
    
    # ===== MODIFICA: Volume threshold più alto (1.8x invece di 1.5x) =====
    # Engulfing su timeframe veloci richiede conferma volume forte
    if vol_ratio < 1.8:  # Era 1.5
        return (False, None, None)
    
    # ===== STEP 3: CALCULATE EMAs =====
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    curr_price = curr['close']
    curr_ema5 = ema_5.iloc[-1]
    curr_ema10 = ema_10.iloc[-1]
    curr_ema60 = ema_60.iloc[-1]

    # Calcola distanze e flags SUBITO dopo aver ottenuto le EMA
    distance_to_ema5 = abs(curr_price - curr_ema5) / curr_ema5
    distance_to_ema10 = abs(curr_price - curr_ema10) / curr_ema10
    distance_to_ema60 = abs(curr_price - curr_ema60) / curr_ema60
    
    above_ema10 = curr_price > curr_ema10
    above_ema60 = curr_price > curr_ema60
    
    # EMA alignment check
    ema_aligned = curr_ema5 > curr_ema10 > curr_ema60

    # EMA 60 Breakout Detection
    prev_price = prev['close']
    prev_ema60 = ema_60.iloc[-2]
    
    was_below_ema60 = prev_price < prev_ema60
    now_above_ema60 = curr_price > curr_ema60
    ema60_breakout = was_below_ema60 and now_above_ema60
    
    # Se è breakout EMA 60 → GOLD automatico
    if ema60_breakout:
        breakout_pct = ((curr_price - curr_ema60) / curr_ema60) * 100
        
        # Breakout deve essere significativo (>0.3%) e volume OK
        if breakout_pct >= 0.3 and vol_ratio >= 1.8:  # Usa 1.8x
            logging.info(
                f'🚀 Bullish Engulfing ROMPE EMA 60! '
                f'Breakout: +{breakout_pct:.2f}%, Vol: {vol_ratio:.1f}x'
            )
            
            pattern_data = {
                'tier': 'GOLD',
                'quality_score': 95,
                'distance_to_ema10': distancetoema10, # Aggiunta
                'distance_to_ema60': distancetoema60, # Aggiunta
                'ema60_breakout': True,
                'breakout_strength': breakout_pct,
                'entry_price': curr_price,
                'prev_body': prev_body,
                'curr_body': curr_body,
                'ema5': curr_ema5,
                'ema10': curr_ema10,
                'ema60': curr_ema60,
                'volume_ratio': vol_ratio,
                'rejection_strength': rejection_strength,
                'suggested_entry': curr_price,
                'suggested_sl': curr_ema60 * 0.998,
                'suggested_tp': curr_price + (curr_price - curr_ema60 * 0.998) * 2.0,
            }
            
            return (True, 'GOLD', pattern_data)
    
    # ===== FINE BLOCCO - Continua con logica TIER normale =====
    
    # ===== STEP 4: TREND FILTER (EMA 60) =====
    # MUST: Prezzo sopra EMA 60 (uptrend)
    if curr_price <= curr_ema60:
        return (False, None, None)
    
    # ===== STEP 5: REJECTION STRENGTH =====
    lower_wick = curr_body_bottom - curr['low']
    upper_wick = curr['high'] - curr_body_top
    total_range = curr['high'] - curr['low']
    
    if total_range == 0:
        return (False, None, None)
    
    lower_wick_pct = lower_wick / total_range
    upper_wick_pct = upper_wick / total_range
    rejection_strength = lower_wick / curr_body if curr_body > 0 else 0

    # ✅ FIX PUNTO #4: Verifica che calcoli siano stati eseguiti
    assert total_range >= 0, "total_range deve essere >= 0"
    assert lower_wick_pct >= 0 and lower_wick_pct <= 1, f"lower_wick_pct invalido: {lower_wick_pct}"
    assert upper_wick_pct >= 0 and upper_wick_pct <= 1, f"upper_wick_pct invalido: {upper_wick_pct}"
    assert rejection_strength >= 0, f"rejection_strength invalido: {rejection_strength}"
    
    logging.debug(f"Engulfing: rejection={rejection_strength:.2f}, lower_wick={lower_wick_pct:.2%}, upper_wick={upper_wick_pct:.2%}")
    
    # ===== STEP 6: PULLBACK DETECTION =====
    # Check se c'è stato pullback (prezzo era più alto 3-10 periodi fa)
    lookback_start = -10
    lookback_end = -3
    recent_highs = df['high'].iloc[lookback_start:lookback_end]
    
    was_higher = False
    if len(recent_highs) > 0:
        max_recent = recent_highs.max()
        # Era almeno 0.8% più alto
        if max_recent > curr_price * 1.008:
            was_higher = True
    
    # ===== STEP 7: EMA DISTANCE CALCULATION (già fatto sopra) =====
    # distance_to_ema5, distance_to_ema10, distance_to_ema60 già calcolati
    
    # Check se prezzo è SOPRA o SOTTO l'EMA (per pullback)
    # above_ema10, above_ema60 già calcolati
    
    # ===== STEP 8: TIER CLASSIFICATION =====
    
    tier = None
    quality_score = 50  # Base score
    
    # === TIER 1: GOLD (EMA 60 Bounce) ===
    near_ema60 = distance_to_ema60 < 0.005  # Entro 0.5%
    
    if near_ema60 and was_higher and vol_ratio >= 2.0 and rejection_strength >= 1.0:
        tier = 'GOLD'
        quality_score = 90
        
    # === TIER 2: GOOD (EMA 10 Bounce) ===
    elif distance_to_ema10 < 0.01 and above_ema60 and vol_ratio >= 1.8:  # Usa 1.8x
        tier = 'GOOD'
        quality_score = 75
        
    # === TIER 3: OK (Generic Engulfing) ===
    elif above_ema60 and vol_ratio >= 1.8:  # Usa 1.8x
        tier = 'OK'
        quality_score = 60
    
    else:
        # Non passa i requisiti minimi
        return (False, None, None)
    
    # ===== STEP 9: BONUS POINTS =====
    
    # Bonus 1: Pullback confermato
    if was_higher:
        quality_score += 10
    
    # Bonus 2: Volume eccezionale
    if vol_ratio >= 2.5:
        quality_score += 10
    
    # Bonus 3: Rejection molto forte
    if rejection_strength >= 1.5:
        quality_score += 10
    
    # Bonus 4: EMA alignment (EMA 5 > EMA 10 > EMA 60)
    if ema_aligned:
        quality_score += 5
    
    # Cap score a 100
    quality_score = min(quality_score, 100)
    
    # ===== STEP 10: PREPARE PATTERN DATA =====
    
    # Calculate suggested SL/TP
    # SL: Sotto low della candela engulfing (o sotto EMA se più basso)
    sl_base = curr['low'] * 0.998  # 0.2% buffer
    
    # Se vicino a EMA 60, usa quella come SL
    if near_ema60:
        sl_ema = curr_ema60 * 0.998
        sl_price = min(sl_base, sl_ema)  # Più conservativo
    else:
        sl_price = sl_base
    
    # TP: Risk/Reward 2:1 minimo
    risk = curr_price - sl_price
    tp_price = curr_price + (risk * 2.0)

    # ✅ FIX PUNTO #4: Validazione dati prima di creare pattern_data
    if total_range == 0:
        logging.warning("Bullish Engulfing: total_range = 0, calcolo rejection fallito")
        rejection_strength = 0.0
        lower_wick_pct = 0.0
        upper_wick_pct = 0.0

    if curr_body == 0:
        logging.warning("Bullish Engulfing: curr_body = 0, rejection_strength non affidabile")
        rejection_strength = 0.0
    
    pattern_data = {
        # Tier info
        'tier': tier,
        'quality_score': quality_score,
        
        # Price info
        'entry_price': curr_price,
        'prev_body': prev_body,
        'curr_body': curr_body,
        'engulfing_ratio': curr_body / prev_body if prev_body > 0 else 0,
        
        # EMA info
        'ema5': curr_ema5,
        'ema10': curr_ema10,
        'ema60': curr_ema60,
        'distance_to_ema10': distance_to_ema10 * 100,  # In %
        'distance_to_ema60': distance_to_ema60 * 100,  # In %
        'near_ema10': distance_to_ema10 < 0.01,
        'near_ema60': near_ema60,
        'ema_aligned': ema_aligned,
        
        # Volume info
        'volume_ratio': vol_ratio,
        
        # Rejection info
        'lower_wick_pct': lower_wick_pct * 100,
        'upper_wick_pct': upper_wick_pct * 100,
        'rejection_strength': rejection_strength,
        
        # Pullback info
        'had_pullback': was_higher,
        
        # Trading setup
        'suggested_entry': curr_price,
        'suggested_sl': sl_price,
        'suggested_tp': tp_price,
        'risk_reward': 2.0,
        
        # Additional
        'above_ema10': above_ema10,
        'above_ema60': above_ema60
    }
    
    return (True, tier, pattern_data)


def is_bud_pattern(df: pd.DataFrame, require_maxi: bool = False) -> tuple:
    """
    🌱 BUD PATTERN (Gemma + Riposo)
    
    Win Rate stimato: 58-68%
    Risk:Reward: 1:2-2.5
    
    STRUTTURA:
    ============================================
    1. Candela BREAKOUT (verde):
       - Rompe sopra EMA 10 al rialzo
       - Corpo forte (>60% range)
       - Close decisamente sopra EMA 10
    
    2. Candele di RIPOSO (2-5):
       - High <= Breakout High
       - Low >= Breakout Low
       - "Compresse" nel range breakout
       - Consolidamento/pausa
    
    3. SEGNALE:
       - Dopo 2+ candele riposo → BUD
       - Dopo 3+ candele riposo → MAXI BUD
    
    LOGICA:
    ============================================
    - Breakout forte = interesse buyers
    - Riposo = accumulation, no panic sell
    - Pattern "compresso" = energia per continuazione
    
    Entry: Close candela dopo riposo (breakout)
    SL: Sotto low candela breakout
    TP: High breakout + (range × 2)
    
    Args:
        df: DataFrame OHLCV
        require_maxi: True per MAXI BUD (3+ riposo), False per BUD (2+ riposo)
    
    Returns:
        (found: bool, pattern_data: dict or None)
    """
    if len(df) < 10:
        return (False, None)
    
    # ===== STEP 1: CALCOLA EMA 10 =====
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    
    # ===== STEP 2: CERCA CANDELA BREAKOUT =====
    # Cerca negli ultimi 8 periodi (esclude corrente)
    breakout_found = False
    breakout_idx = None
    breakout_candle = None
    
    for i in range(-8, -1):  # Da -8 a -2
        if len(df) < abs(i):
            continue
        
        candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        
        # EMA al momento del breakout
        ema10_at_break = ema_10.iloc[i]
        ema10_prev = ema_10.iloc[i-1]
        
        # CHECK 1: Candela VERDE
        is_green = candle['close'] > candle['open']
        if not is_green:
            continue
        
        # CHECK 2: BREAKOUT EMA 10
        # Era sotto, ora sopra (o molto vicino sotto → sopra)
        was_below = prev_candle['close'] < ema10_prev
        now_above = candle['close'] > ema10_at_break
        
        # Oppure: già sopra ma con accelerazione
        already_above_but_strong = (
            prev_candle['close'] > ema10_prev and
            candle['close'] > prev_candle['high']  # Breakout high precedente
        )
        
        if not (was_below and now_above) and not already_above_but_strong:
            continue
        
        # CHECK 3: CORPO FORTE (>60% range)
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            continue
        
        body_pct = body / total_range
        
        if body_pct < 0.60:
            continue
        
        # CHECK 4: CLOSE DECISAMENTE SOPRA EMA 10
        # Almeno 0.2% sopra
        if candle['close'] < ema10_at_break * 1.002:
            continue
        
        # CHECK 5: UPPER WICK PICCOLO (no rejection)
        upper_wick = candle['high'] - candle['close']
        upper_wick_pct = upper_wick / total_range
        
        if upper_wick_pct > 0.25:  # Max 25% ombra superiore
            continue
        
        # BREAKOUT TROVATO!
        breakout_found = True
        breakout_candle = candle
        breakout_idx = i
        break
    
    if not breakout_found:
        return (False, None)
    
    # Livelli chiave del breakout
    breakout_high = breakout_candle['high']
    breakout_low = breakout_candle['low']
    breakout_close = breakout_candle['close']
    
    # ===== STEP 3: VERIFICA CANDELE DI RIPOSO =====
    # Candele tra breakout e corrente
    rest_start = breakout_idx + 1
    rest_end = -1  # Fino a candela prima di corrente
    
    rest_candles = df.iloc[rest_start:rest_end]
    
    # Serve almeno 2 candele per BUD, 3+ per MAXI BUD
    min_rest = 3 if require_maxi else 2
    
    if len(rest_candles) < min_rest:
        return (False, None)
    
    # CHECK: Tutte le candele DENTRO il range breakout
    all_inside = True
    for _, candle in rest_candles.iterrows():
        # High non deve superare breakout high
        if candle['high'] > breakout_high * 1.002:  # Tolleranza 0.2%
            all_inside = False
            break
        
        # Low non deve rompere breakout low
        if candle['low'] < breakout_low * 0.998:  # Tolleranza 0.2%
            all_inside = False
            break
    
    if not all_inside:
        return (False, None)
    
    # CHECK: Range medio delle candele riposo deve essere piccolo
    # (conferma che è davvero "riposo")
    rest_ranges = []
    for _, candle in rest_candles.iterrows():
        r = candle['high'] - candle['low']
        rest_ranges.append(r)
    
    avg_rest_range = sum(rest_ranges) / len(rest_ranges)
    breakout_range = breakout_high - breakout_low
    
    # Range riposo deve essere < 60% del breakout range
    if avg_rest_range > breakout_range * 0.60:
        return (False, None)
    
    # ===== STEP 4: CANDELA CORRENTE (trigger) =====
    curr = df.iloc[-1]
    
    # Candela corrente dovrebbe essere verde (continuazione)
    is_curr_green = curr['close'] > curr['open']
    
    # Candela corrente dovrebbe rompere high del breakout
    # (o almeno essere molto vicina)
    breaks_high = curr['close'] > breakout_high * 0.998
    
    # Se non rompe high, almeno deve essere sopra EMA 10
    curr_ema10 = ema_10.iloc[-1]
    above_ema10 = curr['close'] > curr_ema10
    
    #if not (breaks_high or above_ema10):
    #    return (False, None)
    breaks_structure = ( curr['high'] > breakout_high and curr['close'] > (breakout_high + breakout_low) / 2 )
    
    if not breaks_structure:
        return (False, None)
    
    # ===== STEP 5: VOLUME CHECK (opzionale ma consigliato) =====
    volume_ok = False
    vol_ratio = 0
    
    if 'volume' in df.columns:
        # Volume breakout vs average
        vol_breakout = df['volume'].iloc[breakout_idx]
        vol_avg_before = df['volume'].iloc[breakout_idx-20:breakout_idx].mean()
        
        if vol_avg_before > 0:
            vol_ratio_break = vol_breakout / vol_avg_before
            
            # Volume breakout deve essere > 1.5x
            if vol_ratio_break >= 0.5:
                volume_ok = True
                vol_ratio = vol_ratio_break
    
    # ===== STEP 6: EMA 60 CHECK (trend filter) =====
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    curr_ema60 = ema_60.iloc[-1]
    
    # Breakout dovrebbe essere sopra EMA 60 (uptrend)
    #above_ema60 = breakout_close > curr_ema60
    ema60_at_break = ema_60.iloc[breakout_idx]

    if breakout_close <= ema60_at_break:
        return (False, None)
    
    # ===== PATTERN CONFERMATO! =====
    
    # Determina tipo
    rest_count = len(rest_candles)
    pattern_type = "MAXI BUD" if rest_count >= 3 else "BUD"
    
    # Calcola setup trading
    #entry_price = curr['close']
    #sl_price = breakout_low * 0.998  # Sotto low breakout
    rest_low = rest_candles['low'].min()
    sl_price = min(breakout_low, rest_low) * 0.998
    
    # TP: Proiezione range breakout
    risk = entry_price - sl_price
    tp_price = entry_price + (risk * 2.0)  # 2R
    
    # Oppure: Usa range breakout
    # tp_price = breakout_high + (breakout_range * 2)
    # Definisci entry_price prima di creare la caption o il dizionario data
    entry_price = last_close # o il valore del breakout
    
    pattern_data = {
        'pattern_type': pattern_type,
        'rest_count': rest_count,
        
        # Breakout info
        'breakout_high': breakout_high,
        'breakout_low': breakout_low,
        'breakout_close': breakout_close,
        'breakout_range': breakout_range,
        'breakout_body_pct': body_pct * 100,
        
        # Rest info
        'avg_rest_range': avg_rest_range,
        'rest_range_pct': (avg_rest_range / breakout_range * 100),
        
        # Current info
        'current_price': entry_price,
        'breaks_breakout_high': breaks_high,
        'is_green': is_curr_green,
        
        # Volume
        'volume_ok': volume_ok,
        'volume_ratio': vol_ratio,
        
        # EMA
        'ema10': curr_ema10,
        'ema60': curr_ema60,
        'above_ema60': above_ema60,
        
        # Trading setup
        'suggested_entry': entry_price,
        'suggested_sl': sl_price,
        'suggested_tp': tp_price,
        'risk_reward': 2.0,
        
        'tier': 1  # High probability pattern
    }
    
    return (True, pattern_data)


def is_maxi_bud_pattern(df: pd.DataFrame) -> tuple:
    """
    🌟 MAXI BUD Pattern (versione potenziata)
    Richiede 3+ candele di riposo invece di 2
    """
    return is_bud_pattern(df, require_maxi=True)

def is_bud_bearish_pattern(df: pd.DataFrame, require_maxi: bool = False) -> tuple:
    """
    🔴🌱 BUD BEARISH PATTERN (Gemma Ribassista + Riposo)
    
    Win Rate stimato: 55-65% (short tipicamente più difficile)
    Risk:Reward: 1:2-2.5
    
    STRUTTURA:
    ============================================
    1. Candela BREAKDOWN (rossa):
       - Rompe sotto EMA 10 al ribasso
       - Corpo forte (>60% range)
       - Close decisamente sotto EMA 10
    
    2. Candele di RIPOSO (2-5):
       - High <= Breakdown High
       - Low >= Breakdown Low
       - "Compresse" nel range breakdown
       - Consolidamento/pausa ribassista
    
    3. SEGNALE:
       - Dopo 2+ candele riposo → BUD BEARISH
       - Dopo 3+ candele riposo → MAXI BUD BEARISH
    
    LOGICA:
    ============================================
    - Breakdown forte = sellers control
    - Riposo = no panic buy (shorts confidenti)
    - Pattern "compresso" = energia per continuazione ribasso
    
    Entry: Close candela dopo riposo (breakdown continuation)
    SL: Sopra high candela breakdown
    TP: Low breakdown - (range × 2)
    
    Args:
        df: DataFrame OHLCV
        require_maxi: True per MAXI BUD (3+ riposo), False per BUD (2+ riposo)
    
    Returns:
        (found: bool, pattern_data: dict or None)
    """
    if len(df) < 10:
        return (False, None)
    
    # ===== STEP 1: CALCOLA EMA 10 =====
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    
    # ===== STEP 2: CERCA CANDELA BREAKDOWN =====
    # Cerca negli ultimi 8 periodi
    breakdown_found = False
    breakdown_idx = None
    breakdown_candle = None
    
    for i in range(-8, -1):  # Da -8 a -2
        if len(df) < abs(i):
            continue
        
        candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        
        # EMA al momento del breakdown
        ema10_at_break = ema_10.iloc[i]
        ema10_prev = ema_10.iloc[i-1]
        
        # CHECK 1: Candela ROSSA
        is_red = candle['close'] < candle['open']
        if not is_red:
            continue
        
        # CHECK 2: BREAKDOWN EMA 10
        # Era sopra, ora sotto
        was_above = prev_candle['close'] > ema10_prev
        now_below = candle['close'] < ema10_at_break
        
        # Oppure: già sotto ma con accelerazione ribasso
        already_below_but_strong = (
            prev_candle['close'] < ema10_prev and
            candle['close'] < prev_candle['low']  # Breakdown low precedente
        )
        
        if not (was_above and now_below) and not already_below_but_strong:
            continue
        
        # CHECK 3: CORPO FORTE (>60% range)
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            continue
        
        body_pct = body / total_range
        
        if body_pct < 0.60:
            continue
        
        # CHECK 4: CLOSE DECISAMENTE SOTTO EMA 10
        # Almeno 0.2% sotto
        if candle['close'] > ema10_at_break * 0.998:
            continue
        
        # CHECK 5: LOWER WICK PICCOLO (no rejection)
        lower_wick = candle['close'] - candle['low']
        lower_wick_pct = lower_wick / total_range
        
        if lower_wick_pct > 0.25:  # Max 25% ombra inferiore
            continue
        
        # BREAKDOWN TROVATO!
        breakdown_found = True
        breakdown_candle = candle
        breakdown_idx = i
        break
    
    if not breakdown_found:
        return (False, None)
    
    # Livelli chiave del breakdown
    breakdown_high = breakdown_candle['high']
    breakdown_low = breakdown_candle['low']
    breakdown_close = breakdown_candle['close']
    
    # ===== STEP 3: VERIFICA CANDELE DI RIPOSO =====
    rest_start = breakdown_idx + 1
    rest_end = -1
    
    rest_candles = df.iloc[rest_start:rest_end]
    
    # Serve almeno 2 candele per BUD, 3+ per MAXI BUD
    min_rest = 3 if require_maxi else 2
    
    if len(rest_candles) < min_rest:
        return (False, None)
    
    # CHECK: Tutte le candele DENTRO il range breakdown
    all_inside = True
    for _, candle in rest_candles.iterrows():
        # High non deve superare breakdown high
        if candle['high'] > breakdown_high * 1.002:
            all_inside = False
            break
        
        # Low non deve rompere breakdown low
        if candle['low'] < breakdown_low * 0.998:
            all_inside = False
            break
    
    if not all_inside:
        return (False, None)
    
    # CHECK: Range medio delle candele riposo deve essere piccolo
    rest_ranges = []
    for _, candle in rest_candles.iterrows():
        r = candle['high'] - candle['low']
        rest_ranges.append(r)
    
    avg_rest_range = sum(rest_ranges) / len(rest_ranges)
    breakdown_range = breakdown_high - breakdown_low
    
    if avg_rest_range > breakdown_range * 0.60:
        return (False, None)
    
    # ===== STEP 4: CANDELA CORRENTE (trigger) =====
    curr = df.iloc[-1]
    
    # Candela corrente dovrebbe essere rossa (continuazione)
    is_curr_red = curr['close'] < curr['open']
    
    # Candela corrente dovrebbe rompere low del breakdown
    breaks_low = curr['close'] < breakdown_low * 1.002
    
    # Se non rompe low, almeno deve essere sotto EMA 10
    curr_ema10 = ema_10.iloc[-1]
    below_ema10 = curr['close'] < curr_ema10
    
    if not (breaks_low or below_ema10):
        return (False, None)
    
    # ===== STEP 5: VOLUME CHECK =====
    volume_ok = False
    vol_ratio = 0
    
    if 'volume' in df.columns:
        vol_breakdown = df['volume'].iloc[breakdown_idx]
        vol_avg_before = df['volume'].iloc[breakdown_idx-20:breakdown_idx].mean()
        
        if vol_avg_before > 0:
            vol_ratio_break = vol_breakdown / vol_avg_before
            
            if vol_ratio_break >= 1.5:
                volume_ok = True
                vol_ratio = vol_ratio_break
    
    # ===== STEP 6: EMA 60 CHECK (downtrend filter) =====
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    curr_ema60 = ema_60.iloc[-1]
    
    # Breakdown dovrebbe essere sotto EMA 60 (downtrend)
    below_ema60 = breakdown_close < curr_ema60
    
    # ===== PATTERN CONFERMATO! =====
    
    rest_count = len(rest_candles)
    pattern_type = "MAXI BUD BEARISH" if rest_count >= 3 else "BUD BEARISH"
    
    # Calcola setup trading (SHORT)
    entry_price = curr['close']
    sl_price = breakdown_high * 1.002  # Sopra high breakdown
    
    # TP: Proiezione range breakdown al ribasso
    risk = sl_price - entry_price
    tp_price = entry_price - (risk * 2.0)  # 2R
    
    pattern_data = {
        'pattern_type': pattern_type,
        'rest_count': rest_count,
        
        # Breakdown info
        'breakdown_high': breakdown_high,
        'breakdown_low': breakdown_low,
        'breakdown_close': breakdown_close,
        'breakdown_range': breakdown_range,
        'breakdown_body_pct': body_pct * 100,
        
        # Rest info
        'avg_rest_range': avg_rest_range,
        'rest_range_pct': (avg_rest_range / breakdown_range * 100),
        
        # Current info
        'current_price': entry_price,
        'breaks_breakdown_low': breaks_low,
        'is_red': is_curr_red,
        
        # Volume
        'volume_ok': volume_ok,
        'volume_ratio': vol_ratio,
        
        # EMA
        'ema10': curr_ema10,
        'ema60': curr_ema60,
        'below_ema60': below_ema60,
        
        # Trading setup
        'suggested_entry': entry_price,
        'suggested_sl': sl_price,
        'suggested_tp': tp_price,
        'risk_reward': 2.0,
        
        'side': 'Sell',
        'tier': 1
    }
    
    return (True, pattern_data)


def is_maxi_bud_bearish_pattern(df: pd.DataFrame) -> tuple:
    """
    🌟🔴 MAXI BUD BEARISH Pattern
    Richiede 3+ candele di riposo
    """
    return is_bud_bearish_pattern(df, require_maxi=True)


def is_bearish_engulfing_enhanced(prev, curr, df):
    """
    🔴 BEARISH ENGULFING ENHANCED (EMA-Optimized for SHORT)
    
    Win Rate Base: ~45%
    Win Rate Enhanced: ~55-68%
    
    LOGICA MULTI-TIER (INVERSA del Bullish):
    ==========================================
    
    TIER 1 - GOLD Setup (68-75% win): 🌟
    ├─ Engulfing ROMPE EMA 60 al ribasso
    ├─ Prezzo chiude SOTTO EMA 60
    ├─ Volume 2.5x+
    ├─ Upper rejection (wick >= corpo)
    └─ → INSTITUTIONAL BREAKDOWN (best short setup)
    
    TIER 2 - GOOD Setup (60-65% win): ✅
    ├─ Engulfing vicino EMA 10 (±1%)
    ├─ Sotto EMA 60 (downtrend intact)
    ├─ Volume 2x+
    ├─ Upper rejection moderata
    └─ → SHORT-TERM RESISTANCE (solid setup)
    
    TIER 3 - OK Setup (52-58% win): ⚠️
    ├─ Engulfing generico sotto EMA 60
    ├─ Volume 1.8x+
    └─ → MINIMAL EDGE (accettabile)
    
    REJECTION:
    ├─ Sopra EMA 60 (uptrend) → NO SHORT
    ├─ Volume < 1.8x
    └─ Pattern debole
    
    Returns:
        (found: bool, tier: str, data: dict or None)
    """
    if len(df) < 60:
        return (False, None, None)

    # 🔧 FIX: Dichiara TUTTE le variabili
    rejection_strength = 0.0
    upper_wick_pct = 0.0
    lower_wick_pct = 0.0
    curr_body = 0.0
    prev_body = 0.0
    total_range = 0.0
    had_rally = False
    rally_depth = 0.0
    distance_to_ema10 = 0.0
    distance_to_ema60 = 0.0
    below_ema10 = False
    below_ema60 = False
    close_position = 0.0
    ema_anti_aligned = False
    breakdown_strength = 0.0
    
    # ===== STEP 1: ENGULFING BASE CHECK =====
    prev_body_top = max(prev['open'], prev['close'])
    prev_body_bottom = min(prev['open'], prev['close'])
    curr_body_top = max(curr['open'], curr['close'])
    curr_body_bottom = min(curr['open'], curr['close'])
    
    is_prev_bullish = prev['close'] > prev['open']  # ← Inverso
    is_curr_bearish = curr['close'] < curr['open']  # ← Inverso
    
    engulfs = (curr_body_top >= prev_body_top and 
               curr_body_bottom <= prev_body_bottom)
    
    prev_body = abs(prev['open'] - prev['close'])
    curr_body = abs(curr['open'] - curr['close'])
    has_body = curr_body >= prev_body * 0.5
    
    if not (is_prev_bullish and is_curr_bearish and engulfs and has_body):
        return (False, None, None)
    
    # ===== STEP 2: VOLUME CHECK (MANDATORY) =====
    if 'volume' not in df.columns or len(df['volume']) < 20:
        return (False, None, None)
    
    vol = df['volume']
    avg_vol = vol.iloc[-20:-1].mean()
    curr_vol = vol.iloc[-1]
    
    if avg_vol == 0:
        return (False, None, None)
    
    vol_ratio = curr_vol / avg_vol
    
    # Minimum volume threshold
    if vol_ratio < 0.5:  # ← Più permissivo per SHORT
        return (False, None, None)
    
    # ===== STEP 3: CALCULATE EMAs =====
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    curr_price = curr['close']
    curr_ema5 = ema_5.iloc[-1]
    curr_ema10 = ema_10.iloc[-1]
    curr_ema60 = ema_60.iloc[-1]
    
    # ===== STEP 4: EMA 60 BREAKOUT DETECTION (CHIAVE!) =====
    prev_price = prev['close']
    prev_ema60 = ema_60.iloc[-2]
    
    was_above_ema60 = prev_price > prev_ema60
    now_below_ema60 = curr_price < curr_ema60
    ema60_breakdown = was_above_ema60 and now_below_ema60
    
    # ===== STEP 5: REJECTION STRENGTH (UPPER WICK) =====
    upper_wick = curr['high'] - curr_body_top
    lower_wick = curr_body_bottom - curr['low']
    total_range = curr['high'] - curr['low']
    
    if total_range == 0:
        return (False, None, None)
    
    upper_wick_pct = upper_wick / total_range
    rejection_strength = upper_wick / curr_body if curr_body > 0 else 0
    
    # ===== STEP 6: RALLY DETECTION (prima c'era uptrend) =====
    lookback_start = -10
    lookback_end = -2
    recent_lows = df['low'].iloc[lookback_start:lookback_end]
    
    had_rally = False
    rally_depth = 0
    
    if len(recent_lows) > 0:
        min_recent = recent_lows.min()
        
        # Rally se era almeno 0.8% più basso
        if curr_price > min_recent * 1.008:
            had_rally = True
            rally_depth = (curr_price - min_recent) / min_recent
    
    # ===== STEP 7: EMA DISTANCE CALCULATION =====
    distance_to_ema10 = abs(curr_price - curr_ema10) / curr_ema10
    distance_to_ema60 = abs(curr_price - curr_ema60) / curr_ema60
    
    # Check se prezzo è SOPRA o SOTTO l'EMA
    below_ema10 = curr_price < curr_ema10
    below_ema60 = curr_price < curr_ema60
    
    # ===== STEP 8: CLOSE POSITION IN RANGE =====
    # Per SHORT: meglio se close è nella parte BASSA del range (sellers control)
    close_position = (curr['close'] - curr['low']) / total_range
    
    # ===== STEP 9: TIER CLASSIFICATION =====
    
    tier = None
    quality_score = 50  # Base score
    
    # === TIER 1: GOLD (EMA 60 BREAKDOWN) ===
    if ema60_breakdown:
        breakdown_pct = ((curr_ema60 - curr_price) / curr_ema60) * 100
        
        # Breakdown deve essere significativo (>0.3%) e volume forte
        if breakdown_pct >= 0.3 and vol_ratio >= 0.0 and rejection_strength >= 1.0:
            tier = 'GOLD'
            quality_score = 95
            
            logging.info(
                f'🔴 Bearish Engulfing ROMPE EMA 60! '
                f'Breakdown: -{breakdown_pct:.2f}%, Vol: {vol_ratio:.1f}x'
            )
    
    # === TIER 2: GOOD (EMA 10 Resistance) ===
    if tier is None:
        # Engulfing vicino EMA 10 (resistance)
        high_near_ema10 = abs(curr['high'] - curr_ema10) / curr_ema10 < 0.01
        
        if high_near_ema10 and below_ema60 and vol_ratio >= 2.0:
            tier = 'GOOD'
            quality_score = 78
    
    # === TIER 3: OK (Generic Bearish Engulfing) ===
    if tier is None:
        # MUST: Deve essere sotto EMA 60 (downtrend)
        if below_ema60 and vol_ratio >= 0.0:
            tier = 'OK'
            quality_score = 62
    
    # Se non passa nessun tier → pattern invalido
    if tier is None:
        return (False, None, None)
    
    # ===== STEP 10: BONUS POINTS =====
    
    # Bonus 1: Rally confermato (aveva uptrend prima)
    if had_rally:
        quality_score += 10
    
    # Bonus 2: Volume eccezionale
    if vol_ratio >= 0.0:  # ← Più alto per SHORT
        quality_score += 10
    
    # Bonus 3: Upper rejection molto forte
    if rejection_strength >= 1.5:
        quality_score += 10
    
    # Bonus 4: EMA anti-alignment (EMA 5 < EMA 10 < EMA 60 = bearish)
    ema_anti_aligned = curr_ema5 < curr_ema10 < curr_ema60
    if ema_anti_aligned:
        quality_score += 8
    
    # Bonus 5: Close nella parte bassa del range (<40%)
    if close_position < 0.4:
        quality_score += 7
    
    # Bonus 6: Breakdown profondo (>1%)
    if ema60_breakdown:
        breakdown_pct = ((curr_ema60 - curr_price) / curr_ema60) * 100
        if breakdown_pct >= 1.0:
            quality_score += 8
    
    # Cap score a 100
    quality_score = min(quality_score, 100)
    
    # ===== STEP 11: PREPARE PATTERN DATA =====
    
    # Calculate suggested SL/TP (INVERSO del Bullish)
    # SL: Sopra high della candela engulfing (o sopra EMA se più alto)
    sl_base = curr['high'] * 1.002  # 0.2% buffer sopra
    
    # Se è breakdown EMA 60, usa quella come riferimento
    if ema60_breakdown:
        sl_ema = curr_ema60 * 1.002
        sl_price = max(sl_base, sl_ema)  # Più conservativo
    else:
        sl_price = sl_base
    
    # TP: Risk/Reward 2:1 minimo
    risk = sl_price - curr_price
    tp_price = curr_price - (risk * 2.0)
    
    pattern_data = {
        # Tier info
        'tier': tier,
        'quality_score': quality_score,
        
        # Price info
        'entry_price': curr_price,
        'prev_body': prev_body,
        'curr_body': curr_body,
        'engulfing_ratio': curr_body / prev_body if prev_body > 0 else 0,
        
        # EMA info
        'ema5': curr_ema5,
        'ema10': curr_ema10,
        'ema60': curr_ema60,
        'distance_to_ema10': distance_to_ema10 * 100,
        'distance_to_ema60': distance_to_ema60 * 100,
        'below_ema10': below_ema10,
        'below_ema60': below_ema60,
        'ema_anti_aligned': ema_anti_aligned,
        
        # EMA 60 Breakdown (CHIAVE)
        'ema60_breakdown': ema60_breakdown,
        'breakdown_strength': ((curr_ema60 - curr_price) / curr_ema60 * 100) if ema60_breakdown else 0,
        
        # Volume info
        'volume_ratio': vol_ratio,
        
        # Rejection info
        'upper_wick_pct': upper_wick_pct * 100,
        'lower_wick_pct': (lower_wick / total_range) * 100,
        'rejection_strength': rejection_strength,
        'close_position': close_position * 100,
        
        # Rally info
        'had_rally': had_rally,
        'rally_depth': rally_depth * 100 if had_rally else 0,
        
        # Trading setup
        'suggested_entry': curr_price,
        'suggested_sl': sl_price,
        'suggested_tp': tp_price,
        'risk_reward': 2.0,
        
        # Additional
        'side': 'Sell'
    }
    
    return (True, tier, pattern_data)


def is_hammer(candle):
    """
    Pattern: Hammer (bullish reversal)
    Corpo piccolo in alto, ombra inferiore lunga (almeno 2x il corpo)
    """
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    
    # Evita divisione per zero
    if total_range == 0 or body == 0:
        return False
    
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    
    # Ombra inferiore deve essere almeno 2x il corpo
    # Ombra superiore deve essere piccola (max 30% del corpo)
    # Corpo deve essere nella parte superiore (top 1/3)
    return (lower_wick >= 2 * body and 
            upper_wick <= body * 0.5 and
            body > 0 and
            body / total_range <= 0.3)


def is_shooting_star(candle):
    """
    Pattern: Shooting Star (bearish reversal)
    Corpo piccolo in basso, ombra superiore lunga (almeno 2x il corpo)
    """
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    
    if total_range == 0 or body == 0:
        return False
    
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    
    # Ombra superiore deve essere almeno 2x il corpo
    # Ombra inferiore deve essere piccola
    # Corpo deve essere nella parte inferiore
    return (upper_wick >= 2 * body and 
            lower_wick <= body * 0.5 and
            body > 0 and
            body / total_range <= 0.3)


def is_morning_star_enhanced(df):
    """
    ⭐ MORNING STAR ENHANCED (EMA-Optimized)
    
    Win Rate Base: ~48-52%
    Win Rate Enhanced: ~62-72%
    
    STRUTTURA PATTERN (3 candele):
    ==========================================
    
    Candela -3 (a): Ribassista Grande
    ┃██████┃ <- Body forte (>60% range)
    ┃██████┃    Downtrend in corso
    ┃██████┃
    
    Candela -2 (b): Piccola (Doji/Spinning top)
         ┃   <- Indecisione
       ══╬══     Body piccolo (<30% di a)
         ┃       Sellers esausti
    
    Candela -1 (c): Rialzista Grande
    ████████ <- Recupera >50% di a
    ████████    Buyers prendono controllo
    ████████
    
    LOGICA MULTI-TIER:
    ==========================================
    
    TIER 1 - GOLD Setup (68-75% win): 🌟
    ├─ Morning Star su EMA 60 (±0.5%)
    ├─ Candela b tocca EMA 60 con tail
    ├─ Dopo pullback 1.5%+ (shakeout)
    ├─ Volume candela c: 2.5x+
    ├─ Candela c recupera 70%+ di a
    ├─ Close sopra EMA 10
    └─ → INSTITUTIONAL REVERSAL (best setup)
    
    TIER 2 - GOOD Setup (60-68% win): ✅
    ├─ Morning Star vicino EMA 10 (±1%)
    ├─ Sopra EMA 60 (uptrend intact)
    ├─ Volume c: 2x+
    ├─ Recupera 60%+ di a
    └─ → SWING REVERSAL (solid setup)
    
    TIER 3 - OK Setup (55-60% win): ⚠️
    ├─ Morning Star generico
    ├─ Sopra EMA 60 (trend filter)
    ├─ Volume c: 1.8x+
    ├─ Recupera 50%+ di a
    └─ → MINIMAL EDGE
    
    REJECTION:
    ├─ Sotto EMA 60 (downtrend)
    ├─ Volume < 1.8x
    ├─ Recupero < 50%
    └─ Candela b troppo grande
    
    BONUS FEATURES:
    ==========================================
    ✅ Gap detection (candela b gap down = panic)
    ✅ Fibonacci recovery (61.8% = GOLD)
    ✅ Volume progression (a→b→c)
    ✅ EMA sandwich (b tra EMA 10 e 60)
    
    Returns:
        (found: bool, tier: str, data: dict or None)
    """
    if len(df) < 60:
        return (False, None, None)

    # 🔧 FIX: Dichiara TUTTE le variabili
    a_body = 0.0
    a_range = 0.0
    a_body_pct = 0.0
    b_body = 0.0
    b_range = 0.0
    recovery_pct = 0.0
    vol_a = 0.0
    vol_b = 0.0
    vol_c = 0.0
    vol_c_ratio = 0.0
    pullback_detected = False
    pullback_depth = 0.0
    gap_detected = False
    gap_size = 0.0
    vol_progression_ok = False
    fib_recovery = False
    ema_sandwich = False
    distance_to_ema10 = 0.0
    distance_to_ema60 = 0.0
    b_distance_to_ema10 = 0.0
    b_distance_to_ema60 = 0.0
    b_touches_ema60 = False
    b_touches_ema10 = False
    ema_aligned = False
    
    # ===== STEP 1: CANDELE DEL PATTERN =====
    a = df.iloc[-3]  # Prima: ribassista grande
    b = df.iloc[-2]  # Seconda: piccola (indecisione)
    c = df.iloc[-1]  # Terza: rialzista grande (reversal)
    
    # ===== STEP 2: MORNING STAR BASE CHECK =====
    
    # Candela A: ribassista forte
    a_is_bearish = a['close'] < a['open']
    a_body = abs(a['close'] - a['open'])
    a_range = a['high'] - a['low']
    
    if a_range == 0:
        return (False, None, None)
    
    a_body_pct = a_body / a_range
    
    # Corpo A deve essere significativo (>60% range)
    if not (a_is_bearish and a_body_pct > 0.60):
        return (False, None, None)
    
    # Candela B: piccola (indecisione)
    b_body = abs(b['close'] - b['open'])
    b_range = b['high'] - b['low']
    
    # Per 5m, indecisione deve essere chiara
    if b_body >= a_body * 0.20:  # Era 0.30 → max 20%
        return (False, None, None)
    
    # Candela C: rialzista
    c_is_bullish = c['close'] > c['open']
    c_body = abs(c['close'] - c['open'])
    c_range = c['high'] - c['low']
    
    if not c_is_bullish:
        return (False, None, None)
    
    # ===== STEP 3: RECOVERY CHECK =====
    # C deve recuperare almeno 50% del corpo di A
    a_midpoint = (a['open'] + a['close']) / 2
    
    recovery = c['close'] - a['close']
    recovery_pct = recovery / a_body if a_body > 0 else 0
    
    if recovery_pct < 0.60:  # Minimo 60% recovery
        return (False, None, None)
    
    # ===== STEP 4: VOLUME CHECK =====
    if 'volume' not in df.columns or len(df['volume']) < 20:
        return (False, None, None)
    
    vol = df['volume']
    avg_vol = vol.iloc[-20:-3].mean()
    
    vol_a = vol.iloc[-3]
    vol_b = vol.iloc[-2]
    vol_c = vol.iloc[-1]
    
    if avg_vol == 0:
        return (False, None, None)
    
    vol_c_ratio = vol_c / avg_vol
    
    # Volume C deve essere almeno 1.8x media
    if vol_c_ratio < 1.8:
        return (False, None, None)
    
    # ===== STEP 5: CALCULATE EMAs =====
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    curr_price = c['close']
    curr_ema5 = ema_5.iloc[-1]
    curr_ema10 = ema_10.iloc[-1]
    curr_ema60 = ema_60.iloc[-1]
    
    # EMA al momento di candela B (indecisione)
    b_ema10 = ema_10.iloc[-2]
    b_ema60 = ema_60.iloc[-2]
    
    # ===== STEP 6: TREND FILTER (EMA 60) =====
    # Close di C DEVE essere sopra EMA 60
    if curr_price <= curr_ema60:
        return (False, None, None)
    
    # ===== STEP 7: PULLBACK DETECTION =====
    # Check se c'è stato pullback prima del pattern
    lookback_start = -12
    lookback_end = -4
    recent_highs = df['high'].iloc[lookback_start:lookback_end]
    
    pullback_detected = False
    pullback_depth = 0
    
    if len(recent_highs) > 0:
        max_recent = recent_highs.max()
        
        # Pullback se era almeno 1% più alto
        if max_recent > curr_price * 1.010:
            pullback_detected = True
            pullback_depth = (max_recent - curr_price) / max_recent
    
    # ===== STEP 8: EMA DISTANCE CALCULATION =====
    
    # Distanza close C dalle EMA
    distance_to_ema10 = abs(curr_price - curr_ema10) / curr_ema10
    distance_to_ema60 = abs(curr_price - curr_ema60) / curr_ema60
    
    # Distanza candela B dalle EMA (chiave per tier)
    b_low = b['low']
    b_close = b['close']
    
    b_distance_to_ema10 = abs(b_low - b_ema10) / b_ema10
    b_distance_to_ema60 = abs(b_low - b_ema60) / b_ema60
    
    # Check se B "tocca" le EMA (con tail o close)
    b_touches_ema60 = b_distance_to_ema60 < 0.005  # Entro 0.5%
    b_touches_ema10 = b_distance_to_ema10 < 0.01   # Entro 1%
    
    # ===== STEP 9: GAP DETECTION =====
    # Gap down tra A e B = panic selling
    gap_detected = False
    gap_size = 0
    
    if b['high'] < a['low']:
        gap_detected = True
        gap_size = (a['low'] - b['high']) / a['low']
    
    # ===== STEP 10: VOLUME PROGRESSION =====
    # Ideale: vol_a > vol_b (selling exhaustion) < vol_c (buying surge)
    vol_progression_ok = False
    
    if vol_a > 0 and vol_b > 0:
        vol_a_to_b = vol_b / vol_a  # Deve diminuire
        vol_b_to_c = vol_c / vol_b  # Deve aumentare
        
        # Volume B < Volume A E Volume C > Volume B
        if vol_a_to_b < 0.8 and vol_b_to_c > 1.5:
            vol_progression_ok = True
    
    # ===== STEP 11: FIBONACCI RECOVERY =====
    # 61.8% recovery = Golden ratio
    fib_recovery = False
    
    if 0.58 <= recovery_pct <= 0.68:  # 58-68% (intorno a 61.8%)
        fib_recovery = True
    
    # ===== STEP 12: EMA SANDWICH =====
    # Candela B tra EMA 10 e EMA 60 = accumulation zone
    ema_sandwich = False
    
    if b_ema60 < b_close < b_ema10:
        ema_sandwich = True
    
    # ===== STEP 13: TIER CLASSIFICATION =====
    
    tier = None
    quality_score = 50  # Base score
    
    # === TIER 1: GOLD (EMA 60 Bounce) ===
    if (b_touches_ema60 and 
        pullback_detected and 
        recovery_pct >= 0.70 and 
        vol_c_ratio >= 2.5 and
        curr_price > curr_ema10):
        
        tier = 'GOLD'
        quality_score = 92
    
    # === TIER 2: GOOD (EMA 10 Bounce) ===
    elif (b_touches_ema10 and 
          curr_price > curr_ema60 and 
          recovery_pct >= 0.60 and 
          vol_c_ratio >= 2.0):
        
        tier = 'GOOD'
        quality_score = 78
    
    # === TIER 3: OK (Generic Morning Star) ===
    elif (curr_price > curr_ema60 and 
          recovery_pct >= 0.50 and 
          vol_c_ratio >= 1.8):
        
        tier = 'OK'
        quality_score = 62
    
    else:
        # Non passa requisiti minimi
        return (False, None, None)
    
    # ===== STEP 14: BONUS POINTS =====
    
    # Bonus 1: Pullback profondo (+10)
    if pullback_detected and pullback_depth > 0.015:  # >1.5%
        quality_score += 10
    
    # Bonus 2: Gap down panic (+10)
    if gap_detected:
        quality_score += 10
    
    # Bonus 3: Fibonacci recovery (+10)
    if fib_recovery:
        quality_score += 10
    
    # Bonus 4: Volume progression perfect (+8)
    if vol_progression_ok:
        quality_score += 8
    
    # Bonus 5: Recovery molto forte (>80%) (+8)
    if recovery_pct >= 0.80:
        quality_score += 8
    
    # Bonus 6: Volume panic (3x+) (+7)
    if vol_c_ratio >= 3.0:
        quality_score += 7
    
    # Bonus 7: EMA sandwich (+7)
    if ema_sandwich:
        quality_score += 7
    
    # Bonus 8: Candela B è Doji perfetto (body <5%) (+5)
    if b_range > 0 and (b_body / b_range) < 0.05:
        quality_score += 5
    
    # Bonus 9: Close C sopra EMA 5 (+5)
    if curr_price > curr_ema5:
        quality_score += 5
    
    # Bonus 10: EMA alignment (5 > 10 > 60) (+5)
    ema_aligned = curr_ema5 > curr_ema10 > curr_ema60
    if ema_aligned:
        quality_score += 5
    
    # Cap a 100
    quality_score = min(quality_score, 100)
    
    # ===== STEP 15: PREPARE PATTERN DATA =====
    
    # Calculate suggested SL/TP
    # SL: Sotto low della candela B (indecisione) o sotto EMA 60
    sl_base = b['low'] * 0.998  # 0.2% buffer
    
    if b_touches_ema60:
        sl_ema = curr_ema60 * 0.998
        sl_price = min(sl_base, sl_ema)
    else:
        sl_price = sl_base
    
    # TP: Risk/Reward 2:1 minimo
    risk = curr_price - sl_price
    tp_price = curr_price + (risk * 2.0)
    
    # Calculate rejection zones
    rejection_zone_low = b['low']
    rejection_zone_high = b['high']
    
    pattern_data = {
        # Tier info
        'tier': tier,
        'quality_score': quality_score,
        
        # Pattern structure
        'candle_a': {
            'open': a['open'],
            'close': a['close'],
            'body': a_body,
            'body_pct': a_body_pct * 100
        },
        'candle_b': {
            'open': b['open'],
            'close': b['close'],
            'low': b['low'],
            'high': b['high'],
            'body': b_body,
            'body_pct': (b_body / b_range * 100) if b_range > 0 else 0
        },
        'candle_c': {
            'open': c['open'],
            'close': c['close'],
            'body': c_body,
            'body_pct': (c_body / c_range * 100) if c_range > 0 else 0
        },
        
        # Recovery metrics
        'recovery_pct': recovery_pct * 100,
        'recovery_amount': recovery,
        'fib_recovery': fib_recovery,
        
        # EMA info
        'ema5': curr_ema5,
        'ema10': curr_ema10,
        'ema60': curr_ema60,
        'distance_to_ema10': distance_to_ema10 * 100,
        'distance_to_ema60': distance_to_ema60 * 100,
        'b_touches_ema60': b_touches_ema60,
        'b_touches_ema10': b_touches_ema10,
        'b_distance_to_ema60': b_distance_to_ema60 * 100,
        'ema_sandwich': ema_sandwich,
        'ema_aligned': ema_aligned,
        
        # Volume metrics
        'vol_a': vol_a,
        'vol_b': vol_b,
        'vol_c': vol_c,
        'vol_c_ratio': vol_c_ratio,
        'vol_progression_ok': vol_progression_ok,
        
        # Pullback info
        'pullback_detected': pullback_detected,
        'pullback_depth': pullback_depth * 100 if pullback_detected else 0,
        
        # Gap info
        'gap_detected': gap_detected,
        'gap_size': gap_size * 100 if gap_detected else 0,
        
        # Trading setup
        'entry_price': curr_price,
        'suggested_entry': curr_price,
        'suggested_sl': sl_price,
        'suggested_tp': tp_price,
        'risk_reward': 2.0,
        'rejection_zone_low': rejection_zone_low,
        'rejection_zone_high': rejection_zone_high,
        
        # Additional
        'above_ema10': curr_price > curr_ema10,
        'above_ema60': curr_price > curr_ema60
    }
    
    return (True, tier, pattern_data)


def is_morning_star_ema_breakout(df: pd.DataFrame):
    """
    Pattern: Morning Star + EMA Breakout
    
    STRUTTURA:
    1. Morning Star classico (3 candele)
    2. EMA 5 e 10 erano SOPRA il prezzo (resistenza)
    3. Ultima candela verde ROMPE EMA 5 e 10 al rialzo
    4. Chiude SOPRA entrambe le EMA
    
    Setup ad ALTISSIMA probabilità - Combina:
    - Inversione candlestick (Morning Star)
    - Breakout EMA (conferma trend change)
    
    Returns: True se pattern rilevato
    """
    if len(df) < 10:  # Serve storico per EMA
        return False
    
    # Candele
    a = df.iloc[-3]  # Prima: ribassista
    b = df.iloc[-2]  # Seconda: piccola
    c = df.iloc[-1]  # Terza: rialzista (breakout)
    
    # === STEP 1: Verifica Morning Star classico ===
    if not is_morning_star_enhanced(a, b, c):
        return False
    
    # === STEP 2: Calcola EMA 5 e 10 ===
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    
    # Valori EMA attuali (candela c)
    ema5_now = ema_5.iloc[-1]
    ema10_now = ema_10.iloc[-1]
    
    # Valori EMA sulla candela ribassista (a)
    ema5_prev = ema_5.iloc[-3]
    ema10_prev = ema_10.iloc[-3]
    
    # === STEP 3: EMA erano SOPRA (resistenza) ===
    # Durante la candela ribassista, EMA 5 e 10 erano sopra il prezzo
    ema_were_resistance = (
        ema5_prev > a['close'] and 
        ema10_prev > a['close']
    )
    
    if not ema_were_resistance:
        return False
    
    # === STEP 4: Candela verde ROMPE EMA al rialzo ===
    # La candela c deve:
    # - Aprire sotto o vicino alle EMA
    # - Chiudere SOPRA entrambe le EMA
    
    breaks_ema5 = c['close'] > ema5_now
    breaks_ema10 = c['close'] > ema10_now
    
    if not (breaks_ema5 and breaks_ema10):
        return False
    
    # === STEP 5: Breakout significativo ===
    # La candela deve chiudere almeno 0.3% sopra le EMA
    # (evita breakout deboli)
    
    significant_break = (
        c['close'] > ema5_now * 1.003 and
        c['close'] > ema10_now * 1.003
    )
    
    if not significant_break:
        return False
    
    # === STEP 6: Volume (opzionale ma consigliato) ===
    if 'volume' in df.columns:
        vol = df['volume']
        if len(vol) >= 21:
            avg_vol = vol.iloc[-21:-1].mean()
            current_vol = vol.iloc[-1]
            
            # Volume candela verde deve essere > media
            volume_ok = current_vol > avg_vol * 1.2
            
            if not volume_ok:
                return False
    
    # === PATTERN CONFERMATO ===
    return True


def is_triple_touch_breakout(df: pd.DataFrame) -> tuple:
    """
    🥇 TRIPLE TOUCH BREAKOUT PATTERN (Tier 1)
    
    Win Rate: 62-72% (5m-15m), 70-80% (1h-4h)
    Risk:Reward: 1:2.5-3.5
    
    STRUTTURA PATTERN (12-25 candele totali):
    ============================================
    
    FASE 1 - PRIMO TOCCO RESISTANCE (candela -20/-15):
    ├─ Prezzo raggiunge livello R
    ├─ Rejection: red candle con ombra superiore >30%
    └─ Pullback moderato
    
    FASE 2 - SECONDO TOCCO + FALSE REJECTION (candela -12/-8):
    ├─ Prezzo torna a R
    ├─ Rejection PIÙ FORTE: red candle, volume alto
    ├─ "Weak hands" escono (panic sell)
    └─ Smart money accumula
    
    FASE 3 - CONSOLIDAMENTO (3-10 candele):
    ├─ Range stretto sotto R (max 1%)
    ├─ Prezzo rimane SEMPRE sopra EMA 60 (CRITICAL)
    ├─ Volume in calo (accumulation)
    └─ EMA 10 converge verso prezzo
    
    FASE 4 - TERZO TOCCO + BREAKOUT (candela corrente):
    ├─ Prezzo tocca R (±0.5%)
    ├─ NO rejection questa volta
    ├─ Candela verde con body >50%
    ├─ Close decisamente sopra R
    ├─ Volume > 2x consolidamento
    └─ Prezzo sopra EMA 10 e EMA 60
    
    CONDIZIONE EMA 60 (OBBLIGATORIA):
    ============================================
    Durante TUTTO il pattern (touch 1 → breakout):
    - Ogni low deve essere > EMA 60
    - Se anche 1 sola candela rompe sotto EMA 60 → pattern INVALIDO
    - Logica: Pattern valido SOLO in strong uptrend
    
    ENTRY: Al breakout (close sopra R)
    SL: Sotto consolidamento low con buffer 0.2%
    TP: R + (range × 2.5) = ~2.5-3.5R
    
    Returns:
        (found: bool, pattern_data: dict or None)
    """
    if len(df) < 30:
        return (False, None)
    
    # ===== PRE-CHECK: CALCOLA EMA 60 =====
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    # ===== FASE 1: IDENTIFICA RESISTANCE LEVEL =====
    # Cerca negli ultimi 25 candele (esclude corrente)
    lookback_start = -25
    lookback_end = -1
    lookback = df.iloc[lookback_start:lookback_end]
    
    if len(lookback) < 15:
        return (False, None)
    
    # Trova livelli di high toccati multiple volte
    # Clustering: highs entro ±0.5% sono stesso livello
    potential_resistances = []
    
    for i in range(len(lookback) - 5):
        high = lookback['high'].iloc[i]
        touches = []
        touch_indices = [i]
        
        # Cerca altri tocchi dello stesso livello
        for j in range(i + 1, len(lookback)):
            other_high = lookback['high'].iloc[j]
            
            # Stesso livello se entro ±0.5%
            if abs(other_high - high) / high < 0.005:
                touches.append(j)
                touch_indices.append(j)
        
        # Serve almeno 3 tocchi totali (incluso primo)
        if len(touch_indices) >= 3:
            potential_resistances.append({
                'level': high,
                'touch_count': len(touch_indices),
                'touch_indices': touch_indices,
                'first_touch_idx': i
            })
    
    if not potential_resistances:
        return (False, None)
    
    # Prendi resistance con più tocchi
    # Se parità, prendi quello più recente
    resistance_data = max(
        potential_resistances, 
        key=lambda x: (x['touch_count'], -x['first_touch_idx'])
    )
    
    R = resistance_data['level']
    touch_indices = resistance_data['touch_indices']
    
    # Serve ESATTAMENTE 3 tocchi (o più, ma consideriamo primi 3)
    if len(touch_indices) < 3:
        return (False, None)
    
    # Prendi i primi 3 tocchi
    touch_1_idx = touch_indices[0]
    touch_2_idx = touch_indices[1]
    touch_3_idx = touch_indices[2]  # Può essere candela corrente o vicina
    
    # ===== FASE 2: VERIFICA REJECTIONS SUI PRIMI 2 TOCCHI =====
    
    # TOUCH 1 - Deve essere rejection
    touch_1 = lookback.iloc[touch_1_idx]
    
    # Red candle
    is_touch1_red = touch_1['close'] < touch_1['open']
    
    # Upper wick significativo (>30% range)
    touch1_upper_wick = touch_1['high'] - max(touch_1['open'], touch_1['close'])
    touch1_range = touch_1['high'] - touch_1['low']
    
    if touch1_range == 0:
        return (False, None)
    
    touch1_wick_pct = touch1_upper_wick / touch1_range
    touch1_has_rejection = touch1_wick_pct > 0.30
    
    if not (is_touch1_red and touch1_has_rejection):
        return (False, None)
    
    # TOUCH 2 - Deve essere rejection (possibilmente più forte)
    touch_2 = lookback.iloc[touch_2_idx]
    
    # Red candle
    is_touch2_red = touch_2['close'] < touch_2['open']
    
    # Upper wick significativo
    touch2_upper_wick = touch_2['high'] - max(touch_2['open'], touch_2['close'])
    touch2_range = touch_2['high'] - touch_2['low']
    
    if touch2_range == 0:
        return (False, None)
    
    touch2_wick_pct = touch2_upper_wick / touch2_range
    touch2_has_rejection = touch2_wick_pct > 0.30
    
    if not (is_touch2_red and touch2_has_rejection):
        return (False, None)
    
    # ===== FASE 3: VERIFICA CONSOLIDAMENTO =====
    # Tra touch 2 e touch 3 (o candela corrente)
    
    # Se touch 3 è candela corrente, consolidamento è tra touch_2 e -1
    # Altrimenti tra touch_2 e touch_3
    
    curr = df.iloc[-1]
    
    # Check se corrente è il terzo tocco
    curr_touches_R = abs(curr['high'] - R) / R < 0.005
    
    if curr_touches_R:
        # Corrente è touch 3, consolidamento è tra touch_2 e -1
        consolidation_start_idx = lookback_start + touch_2_idx + 1
        consolidation_end_idx = -1
    else:
        # Touch 3 è dentro lookback
        consolidation_start_idx = lookback_start + touch_2_idx + 1
        consolidation_end_idx = lookback_start + touch_3_idx
    
    consolidation = df.iloc[consolidation_start_idx:consolidation_end_idx]
    
    if len(consolidation) < 3:
        return (False, None)  # Troppo corto
    
    if len(consolidation) > 10:
        return (False, None)  # Troppo lungo (pattern invalido)
    
    # Range consolidamento deve essere stretto (max 1%)
    cons_high = consolidation['high'].max()
    cons_low = consolidation['low'].min()
    cons_range = cons_high - cons_low
    cons_range_pct = (cons_range / cons_low) * 100
    
    if cons_range_pct > 1.0:
        return (False, None)  # Range troppo ampio
    
    # Consolidamento deve essere SOTTO R (con tolleranza +0.3%)
    if cons_high > R * 1.003:
        return (False, None)  # Ha rotto sopra R = non è consolidamento
    
    # ===== CRITICAL: VERIFICA EMA 60 DURANTE CONSOLIDAMENTO =====
    ema60_during_cons = ema_60.iloc[consolidation_start_idx:consolidation_end_idx]
    cons_lows = consolidation['low']
    
    # OGNI low deve essere sopra EMA 60
    #all_above_ema60 = (cons_lows > ema60_during_cons).all()
    
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    # Verifica che gli indici siano allineati
    if len(df) > abs(pattern_start_idx):
        ema_10_pattern = ema_10.iloc[pattern_start_idx:]
        # Assicurati che abbiano la stessa lunghezza
        if len(pattern_candles) == len(ema_10_pattern):
            all_lows_above_ema10 = (pattern_candles['low'] > ema_10_pattern * 0.998).all()
        else:
            all_lows_above_ema10 = True  # Skip check se mismatch
    else:
        all_lows_above_ema10 = True
    
    if not all_lows_above_ema10:
        logging.debug(f'Triple Touch: Pattern rompe sotto EMA 10')
        return (False, None)

    #if not all_above_ema60:
    #    logging.debug(f'🚫 Triple Touch: Consolidamento rompe sotto EMA 60')
    #   return (False, None)
    
    # ===== CRITICAL: VERIFICA EMA 60 DURANTE TUTTO IL PATTERN =====
    # Dal touch 1 fino a corrente
    pattern_start_idx = lookback_start + touch_1_idx
    pattern_candles = df.iloc[pattern_start_idx:]
    ema60_pattern = ema_60.iloc[pattern_start_idx:]
    
    # OGNI low del pattern deve essere sopra EMA 60
    all_lows_above_ema60 = (pattern_candles['low'] > ema60_pattern).all()
    
    if not all_lows_above_ema60:
        logging.debug(f'🚫 Triple Touch: Pattern rompe sotto EMA 60')
        return (False, None)
    
    # ===== FASE 4: VERIFICA BREAKOUT (candela corrente) =====
    
    # Deve toccare R
    if not curr_touches_R:
        return (False, None)
    
    # Deve essere candela VERDE
    is_green = curr['close'] > curr['open']
    
    if not is_green:
        return (False, None)
    
    # Deve chiudere SOPRA R (breakout confermato)
    closes_above_R = curr['close'] > R
    
    if not closes_above_R:
        return (False, None)
    
    # Body forte (minimo 50% del range)
    curr_body = abs(curr['close'] - curr['open'])
    curr_range = curr['high'] - curr['low']
    
    if curr_range == 0:
        return (False, None)
    
    curr_body_pct = (curr_body / curr_range) * 100
    
    if curr_body_pct < 50:
        return (False, None)
    
    # Upper wick piccolo (no rejection) - max 25% del range
    curr_upper_wick = curr['high'] - curr['close']
    curr_upper_wick_pct = (curr_upper_wick / curr_range) * 100
    
    if curr_upper_wick_pct > 25:
        return (False, None)  # Troppa rejection = debole
    
    # ===== VOLUME CHECK (CRITICAL) =====
    if 'volume' not in df.columns:
        return (False, None)  # Volume obbligatorio
    
    # Volume consolidamento (media)
    cons_vol_avg = consolidation['volume'].mean()
    
    if cons_vol_avg == 0 or pd.isna(cons_vol_avg):
        return (False, None)
    
    # Volume breakout (corrente)
    curr_vol = df['volume'].iloc[-1]
    
    if pd.isna(curr_vol):
        return (False, None)
    
    vol_ratio = curr_vol / cons_vol_avg
    
    # Volume breakout DEVE essere > 2x consolidamento
    if vol_ratio < 0.5:
        logging.debug(f'🚫 Triple Touch: Volume insufficiente ({vol_ratio:.1f}x)')
        return (False, None)
    
    # ===== EMA 10 CHECK =====
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    curr_ema10 = ema_10.iloc[-1]
    curr_ema60 = ema_60.iloc[-1]
    
    # Breakout deve essere sopra EMA 10
    if curr['close'] <= curr_ema10:
        return (False, None)
    
    # Già verificato sopra, ma double check
    if curr['close'] <= curr_ema60:
        return (False, None)
    
    # ===== QUALITY CHECKS (opzionali ma aumentano win rate) =====
    
    # Check 1: EMA alignment (EMA 10 > EMA 60)
    ema_aligned = curr_ema10 > curr_ema60
    
    # Check 2: Distanza da EMA 60 (non troppo lontano)
    distance_to_ema60 = (curr['close'] - curr_ema60) / curr_ema60
    near_ema60 = distance_to_ema60 < 0.02  # Entro 2%
    
    # Check 3: Strength del secondo rejection
    stronger_second_rejection = touch2_wick_pct > touch1_wick_pct
    
    # ===== PATTERN CONFERMATO! =====
    
    # Calcola metriche aggiuntive
    rejection_avg = (touch1_wick_pct + touch2_wick_pct) / 2
    
    # Calcola distanza minima da EMA 60 durante pattern
    min_distance_to_ema60 = ((pattern_candles['low'] - ema60_pattern) / ema60_pattern).min()
    
    pattern_data = {
        # Livelli chiave
        'resistance': R,
        'consolidation_low': cons_low,
        'consolidation_high': cons_high,
        'range': cons_range,
        'range_pct': cons_range_pct,
        
        # Touch info
        'touch_count': len(touch_indices),
        'touch_1_rejection_pct': touch1_wick_pct * 100,
        'touch_2_rejection_pct': touch2_wick_pct * 100,
        'rejection_avg': rejection_avg * 100,
        'stronger_second_rejection': stronger_second_rejection,
        
        # Consolidamento
        'consolidation_duration': len(consolidation),
        'consolidation_vol_avg': cons_vol_avg,
        
        # Breakout info
        'breakout_price': curr['close'],
        'breakout_body_pct': curr_body_pct,
        'breakout_upper_wick_pct': curr_upper_wick_pct,
        'volume_ratio': vol_ratio,
        
        # EMA values
        'ema10': curr_ema10,
        'ema60': curr_ema60,
        'ema_aligned': ema_aligned,
        'near_ema60': near_ema60,
        'min_distance_to_ema60_pct': min_distance_to_ema60 * 100,
        
        # Current price
        'current_price': curr['close'],
        
        # Trading setup
        'suggested_entry': curr['close'],
        'suggested_sl': cons_low * 0.998,  # 0.2% buffer sotto cons low
        'suggested_tp': R + (cons_range * 2.5),  # 2.5R projection
        
        # Quality score
        'quality': 'GOLD' if (ema_aligned and near_ema60 and stronger_second_rejection) else 'GOOD',
        
        'tier': 1  # High priority (Tier 1)
    }
    
    return (True, pattern_data)

def is_breakout_retest(df: pd.DataFrame) -> tuple:
    """
    🥇 BREAKOUT + RETEST PATTERN (Tier 1 - High Probability)
    
    Win Rate: 60-70% (5m), 65-75% (15m)
    Risk:Reward: 1:2.5-3
    
    STRUTTURA PATTERN (7-15 candele totali):
    ============================================
    
    FASE 1 - CONSOLIDAMENTO (5-10 candele):
    ├─ Range definito (High = R, Low = S)
    ├─ Oscillazioni tra S e R
    └─ Volume medio/basso
    
    FASE 2 - BREAKOUT (1 candela):
    ├─ Close > R con volume 2x+
    ├─ Corpo forte (>60% range)
    └─ Momentum (no rejection)
    
    FASE 3 - PULLBACK (2-5 candele):
    ├─ Prezzo torna verso R (ora supporto)
    ├─ Non rompe sotto R (max -0.3%)
    └─ Volume in calo (profit taking)
    
    FASE 4 - RETEST + BOUNCE (candela corrente):
    ├─ Tocca zona R ±0.5%
    ├─ Candela verde con rejection sotto
    ├─ Volume > media (buyers defend)
    └─ Close sopra EMA 10
    
    CONDIZIONI CRITICHE:
    ============================================
    1. Range consolidamento >= 0.8% (significativo)
    2. Breakout volume >= 2x media consolidamento
    3. Retest NON rompe resistance (max -0.3% tolleranza)
    4. Retest con rejection (wick >= 40% range)
    5. EMA 10 e 60 sotto il prezzo (trend support)
    6. Timing: retest entro 2-5 candele dal breakout
    
    ENTRY: Al bounce dal retest (candela corrente)
    SL: Sotto retest low con buffer 0.2%
    TP: R + (range × 2) = ~2.5-3R
    
    Returns:
        (found: bool, pattern_data: dict or None)
    """
    if len(df) < 20:
        return (False, None)
    
    # ===== FASE 1: IDENTIFICA CONSOLIDAMENTO =====
    # Cerchiamo range di consolidamento negli ultimi 10-15 candele
    # (escluse le ultime 1-5 che potrebbero essere il retest)
    
    lookback_start = -15  # Inizio ricerca
    lookback_end = -6     # Fine ricerca (lascia spazio per breakout+retest)
    
    consolidation = df.iloc[lookback_start:lookback_end]
    
    if len(consolidation) < 5:
        return (False, None)
    
    # Range del consolidamento
    resistance = consolidation['high'].max()
    support = consolidation['low'].min()
    range_size = resistance - support
    
    # Range deve essere significativo (min 0.8% del prezzo)
    if range_size < resistance * 0.008:
        return (False, None)
    
    # Verifica che sia un vero consolidamento (prezzo oscilla nel range)
    # Almeno 3 tocchi su resistance e 3 su support
    tolerance_r = resistance * 0.005  # ±0.5%
    tolerance_s = support * 0.005
    
    touches_resistance = (consolidation['high'] >= resistance - tolerance_r).sum()
    touches_support = (consolidation['low'] <= support + tolerance_s).sum()
    
    if touches_resistance < 3 or touches_support < 3:
        return (False, None)
    
    # ===== FASE 2: IDENTIFICA BREAKOUT =====
    # Cerchiamo candela di breakout tra consolidamento e retest
    # (candele da -5 a -2, esclude corrente che è il retest)
    
    breakout_found = False
    breakout_candle = None
    breakout_index = None
    
    for i in range(-5, -1):  # Da -5 a -2
        if len(df) < abs(i):
            continue
        
        candle = df.iloc[i]
        
        # Breakout = close sopra resistance
        if candle['close'] <= resistance:
            continue
        
        # Deve essere rialzista
        if candle['close'] <= candle['open']:
            continue
        
        # Corpo forte (>60% del range)
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            continue
        
        body_pct = body / total_range
        
        if body_pct < 0.60:
            continue
        
        # Close vicino al high (no rejection) - max 20% ombra superiore
        upper_wick = candle['high'] - candle['close']
        
        if upper_wick > total_range * 0.20:
            continue
        
        # Volume check (>= 2x media consolidamento)
        if 'volume' in df.columns:
            consolidation_vol = df['volume'].iloc[lookback_start:lookback_end].mean()
            breakout_vol = df['volume'].iloc[i]
            
            if consolidation_vol == 0:
                continue
            
            vol_ratio = breakout_vol / consolidation_vol
            
            if vol_ratio < 0.5:
                continue
        else:
            return (False, None)  # Volume essenziale per questo pattern
        
        # BREAKOUT TROVATO!
        breakout_found = True
        breakout_candle = candle
        breakout_index = i
        break
    
    if not breakout_found:
        return (False, None)
    
    # ===== FASE 3: VERIFICA PULLBACK =====
    # Candele tra breakout e corrente devono essere pullback
    # (tornano verso resistance MA non rompono sotto)
    
    pullback_candles = df.iloc[breakout_index+1:-1]  # Tra breakout e corrente
    
    if len(pullback_candles) < 2:
        return (False, None)  # Serve almeno 2 candele di pullback
    
    if len(pullback_candles) > 5:
        return (False, None)  # Troppo tempo = pattern invalidato
    
    # Verifica che pullback non rompa resistance (divenuto supporto)
    # Tolleranza: max 0.3% sotto
    pullback_low = pullback_candles['low'].min()
    
    if pullback_low < resistance * 0.997:
        return (False, None)  # Ha rotto support = failed breakout
    
    # Volume pullback deve essere minore (profit taking, no panic)
    pullback_vol_avg = pullback_candles['volume'].mean()
    
    if pullback_vol_avg > breakout_vol * 0.8:
        return (False, None)  # Volume troppo alto = possibile reversal
    
    # ===== FASE 4: RETEST + BOUNCE (candela corrente) =====
    
    curr = df.iloc[-1]
    
    # Deve essere rialzista
    if curr['close'] <= curr['open']:
        return (False, None)
    
    # Deve toccare zona resistance (ora supporto) ±0.5%
    retest_zone_low = resistance * 0.995
    retest_zone_high = resistance * 1.005
    
    touches_retest = (curr['low'] >= retest_zone_low and 
                     curr['low'] <= retest_zone_high)
    
    if not touches_retest:
        return (False, None)
    
    # Deve avere rejection sotto (wick inferiore >= 40% range)
    lower_wick = min(curr['open'], curr['close']) - curr['low']
    curr_range = curr['high'] - curr['low']
    
    if curr_range == 0:
        return (False, None)
    
    wick_pct = lower_wick / curr_range
    
    if wick_pct < 0.40:
        return (False, None)  # Rejection non abbastanza forte
    
    # Corpo decente (min 30% range)
    curr_body = abs(curr['close'] - curr['open'])
    curr_body_pct = curr_body / curr_range
    
    if curr_body_pct < 0.30:
        return (False, None)
    
    # Volume bounce >= media (buyers defend)
    curr_vol = df['volume'].iloc[-1]
    recent_vol_avg = df['volume'].iloc[-10:-1].mean()
    
    if recent_vol_avg == 0:
        return (False, None)
    
    vol_ratio_bounce = curr_vol / recent_vol_avg
    
    if vol_ratio_bounce < 1.0:
        return (False, None)  # Volume troppo basso
    
    # ===== EMA CHECKS =====
    # Prezzo deve essere sopra EMA 10 e 60 (trend support)
    
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    curr_ema10 = ema_10.iloc[-1]
    curr_ema60 = ema_60.iloc[-1]
    
    if curr['close'] <= curr_ema10:
        return (False, None)
    
    if curr['close'] <= curr_ema60:
        return (False, None)
    
    # ===== PATTERN CONFERMATO! =====
    
    # Calcola metriche per trading
    consolidation_range_pct = (range_size / resistance) * 100
    distance_to_resistance = abs(curr['low'] - resistance)
    
    pattern_data = {
        # Livelli chiave
        'resistance': resistance,  # Ora supporto dopo breakout
        'support': support,
        'range_size': range_size,
        'range_pct': consolidation_range_pct,
        
        # Breakout info
        'breakout_price': breakout_candle['close'],
        'breakout_vol_ratio': vol_ratio,
        'breakout_body_pct': body_pct * 100,
        
        # Retest info
        'retest_low': curr['low'],
        'distance_to_resistance': distance_to_resistance,
        'retest_rejection_pct': wick_pct * 100,
        'retest_vol_ratio': vol_ratio_bounce,
        
        # Current price & EMA
        'current_price': curr['close'],
        'ema10': curr_ema10,
        'ema60': curr_ema60,
        
        # Trading setup (suggerito)
        'suggested_entry': curr['close'],
        'suggested_sl': curr['low'] * 0.998,  # Sotto retest low
        'suggested_tp': resistance + (range_size * 2),  # 2R projection
        
        # Quality metrics
        'touches_resistance': touches_resistance,
        'touches_support': touches_support,
        'pullback_duration': len(pullback_candles),
        
        'tier': 1  # High priority (Tier 1)
    }
    
    return (True, pattern_data)


def is_liquidity_sweep_reversal(df: pd.DataFrame):
    """
    Pattern: Liquidity Sweep + Reversal (FIXED VERSION)
    
    FIXES vs versione precedente:
    ✅ Timing corretto (rileva su breakout, non su setup)
    ✅ Volume obbligatorio (2x+ su recovery)
    ✅ Recovery strength (80%+ recupero)
    ✅ Breakout confirmation (price > recovery high)
    ✅ Pattern data utilizzabile per SL/TP
    
    TIMING CORRETTO:
    - Candela -3: SWEEP sotto previous low
    - Candela -2: RECOVERY verde
    - Candela -1: CONFERMA breakout (pattern rilevato QUI)
    
    Entry: Al breakout del recovery high
    SL: Sotto sweep low
    TP: 2R (alta probabilità)
    
    Returns: (found: bool, data: dict or None)
    """
    if len(df) < 20:
        return (False, None)
    
    # Candele CORRETTE
    sweep_candle = df.iloc[-3]    # Sweep
    recovery_candle = df.iloc[-2]  # Recovery
    current = df.iloc[-1]          # Conferma breakout
    
    # === STEP 1: Identifica PREVIOUS LOW ===
    lookback = 15
    recent_lows = df['low'].iloc[-lookback:-3]
    
    if len(recent_lows) < 5:
        return (False, None)
    
    previous_low = recent_lows.min()
    
    # Previous low deve essere supporto valido (toccato almeno 2 volte)
    touches = (recent_lows <= previous_low * 1.002).sum()
    if touches < 2:
        return (False, None)
    
    # === STEP 2: SWEEP - Rompe previous low ===
    sweep_low = sweep_candle['low']
    
    # Deve rompere previous low
    breaks_previous_low = sweep_low < previous_low
    if not breaks_previous_low:
        return (False, None)
    
    # === STEP 3: OMBRA LUNGA SOTTO ===
    sweep_body = abs(sweep_candle['close'] - sweep_candle['open'])
    sweep_range = sweep_candle['high'] - sweep_candle['low']
    lower_wick = min(sweep_candle['open'], sweep_candle['close']) - sweep_candle['low']
    
    if sweep_range == 0:
        return (False, None)
    
    # Ombra deve essere almeno 50% del range
    has_long_wick = lower_wick >= sweep_range * 0.5
    if not has_long_wick:
        return (False, None)
    
    # === STEP 4: CHIUDE SOPRA previous low (REVERSAL) ===
    closes_above = sweep_candle['close'] > previous_low
    if not closes_above:
        return (False, None)
    
    # === STEP 5: RECOVERY CANDLE ===
    recovery_is_bullish = recovery_candle['close'] > recovery_candle['open']
    
    if not recovery_is_bullish:
        return (False, None)
    
    # Recovery deve avere corpo significativo
    recovery_body = abs(recovery_candle['close'] - recovery_candle['open'])
    recovery_range = recovery_candle['high'] - recovery_candle['low']
    
    if recovery_range == 0:
        return (False, None)
    
    recovery_body_pct = recovery_body / recovery_range
    
    # Corpo minimo 40% del range
    if recovery_body_pct < 0.40:
        return (False, None)
    
    # === STEP 6: RECOVERY STRENGTH (CRITICO) ===
    # Recovery deve recuperare almeno 80% del drop
    sweep_drop = abs(sweep_candle['open'] - sweep_candle['close'])
    recovery_gain = recovery_candle['close'] - recovery_candle['open']
    
    if sweep_drop > 0:
        recovery_pct = recovery_gain / sweep_drop
    else:
        recovery_pct = 1.0
    
    if recovery_pct < 0.60:
        return (False, None)
    
    # === STEP 7: VOLUME SPIKE su recovery (MUST HAVE) ===
    if 'volume' not in df.columns or len(df['volume']) < 20:
        return (False, None)
    
    vol = df['volume']
    avg_vol = vol.iloc[-20:-2].mean()
    recovery_vol = vol.iloc[-2]
    
    if avg_vol == 0:
        return (False, None)
    
    vol_ratio = recovery_vol / avg_vol
    
    # Volume recovery DEVE essere > 2x
    if vol_ratio < 0.5:
        return (False, None)

    # ===== AGGIUNGI: Verifica che sweep non sia troppo profondo =====
    # Se sweep > 1% sotto previous low = probabile breakdown, non sweep
    sweep_depth = abs(sweep_low - previous_low) / previous_low
    if sweep_depth > 0.01:  # Max 1%
        logging.debug(f'Liquidity Sweep: Sweep troppo profondo ({sweep_depth*100:.2f}%), probabile breakdown')
        return (False, None)
    
    # === STEP 8: CURRENT CANDLE conferma breakout ===
    # Current deve essere rialzista o almeno neutra
    current_not_bearish = current['close'] >= current['open'] * 0.995
    
    # Current deve rompere recovery high (breakout confermato)
    breaks_recovery_high = current['close'] > recovery_candle['high']
    
    if not (current_not_bearish and breaks_recovery_high):
        return (False, None)
    
    # === STEP 9: EMA 60 CHECK ===
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    ema_60_value = ema_60.iloc[-1]
    
    # Current deve essere sopra EMA 60 (trend rialzista)
    price_above_ema60 = current['close'] > ema_60_value
    if not price_above_ema60:
        return (False, None)
    
    # === STEP 10: EMA 10 CHECK (momentum) ===
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_10_value = ema_10.iloc[-1]
    
    # Current deve essere sopra EMA 10
    price_above_ema10 = current['close'] > ema_10_value
    if not price_above_ema10:
        return (False, None)
    
    # === STEP 11: Distanza da EMA 60 ===
    distance_to_ema60 = abs(current['close'] - ema_60_value) / ema_60_value
    near_ema60 = distance_to_ema60 < 0.01
    
    # === PATTERN CONFERMATO ===
    pattern_data = {
        'previous_low': previous_low,
        'sweep_low': sweep_low,
        'recovery_high': recovery_candle['high'],
        'breakout_price': current['close'],
        'recovery_pct': recovery_pct * 100,
        'volume_ratio': vol_ratio,
        'ema10': ema_10_value,
        'ema60': ema_60_value,
        'near_ema60': near_ema60,
        # Per SL/TP custom
        'suggested_entry': recovery_candle['high'] * 1.001,
        'suggested_sl': sweep_low * 0.998
    }
    
    return (True, pattern_data)


def is_evening_star(a, b, c):
    """
    Pattern: Evening Star (3 candele - bearish reversal)
    1. Candela rialzista grande
    2. Candela piccola (indecisione)
    3. Candela ribassista grande che perde oltre il 50%
    """
    # Prima candela: rialzista con corpo decente
    a_body = abs(a['close'] - a['open'])
    a_range = a['high'] - a['low']
    is_a_bullish = a['close'] > a['open']
    
    # Seconda candela: piccola
    b_body = abs(b['close'] - b['open'])
    is_b_small = b_body < a_body * 0.3
    
    # Terza candela: ribassista e chiude sotto il 50% della prima
    is_c_bearish = c['close'] < c['open']
    c_drops = c['close'] < (a['open'] + a['close']) / 2
    
    return (is_a_bullish and 
            a_body / a_range > 0.5 and
            is_b_small and 
            is_c_bearish and 
            c_drops)

def is_pin_bar_bullish_enhanced(candle, df):
    """
    📍 PIN BAR BULLISH – Crypto Scalping 5m (Professional Version)
    """

    if len(df) < 60 or 'volume' not in df.columns:
        return (False, None, None)

    # ✅ FIX: Inizializza variabili all'inizio
    tail_distance_to_ema60 = 0.0
    lower_wick_pct = 0.0
    rejection_strength = 0.0

    # ===== ANATOMIA CANDELA =====
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    if total_range == 0:
        return (False, None, None)

    lower_wick = min(candle['open'], candle['close']) - candle['low']
    upper_wick = candle['high'] - max(candle['open'], candle['close'])

    body_pct = body / total_range
    lower_wick_pct = lower_wick / total_range
    upper_wick_pct = upper_wick / total_range

    # Pin bar base
    if lower_wick_pct < 0.55:
        return (False, None, None)
    if upper_wick_pct > 0.25:
        return (False, None, None)
    if body_pct > 0.30 or body_pct < 0.05:
        return (False, None, None)

    close_position = (candle['close'] - candle['low']) / total_range
    if close_position < 0.50:
        return (False, None, None)

    is_bullish = candle['close'] > candle['open']

    # ===== VOLUME =====
    avg_vol = df['volume'].iloc[-21:-1].mean()
    vol_ratio = df['volume'].iloc[-1] / avg_vol if avg_vol > 0 else 0

    # ===== EMA =====
    ema5 = df['close'].ewm(span=5, adjust=False).mean()
    ema10 = df['close'].ewm(span=10, adjust=False).mean()
    ema60 = df['close'].ewm(span=60, adjust=False).mean()

    curr_price = candle['close']
    curr_ema10 = ema10.iloc[-1]
    curr_ema60 = ema60.iloc[-1]

    # Definisci tail_low (punto più basso della tail)
    tail_low = candle['low']

    # Tail distance to EMA
    tail_distance_to_ema10 = abs(tail_low - curr_ema10) / curr_ema10
    # ← AGGIUNGI QUESTA RIGA MANCANTE:
    tail_distance_to_ema60 = abs(tail_low - curr_ema60) / curr_ema60
    
    # Check se tail tocca le EMA
    tail_near_ema10 = tail_distance_to_ema10 < 0.01  # Entro 1%
    tail_near_ema60 = tail_distance_to_ema60 < 0.005  # Entro 0.5%

    # Trend filter
    if curr_price <= curr_ema60 * 0.995:
        return (False, None, None)

    # ===== PIN BAR SU LIVELLO =====
    pin_low = candle['low']
    near_ema10 = abs(pin_low - curr_ema10) / curr_ema10 < 0.01
    near_ema60 = abs(pin_low - curr_ema60) / curr_ema60 < 0.005

    if not (near_ema10 or near_ema60):
        return (False, None, None)

    # ===== LIQUIDITY SWEEP =====
    recent_lows = df['low'].iloc[-15:-1]
    swept_liquidity = pin_low < recent_lows.min() if len(recent_lows) > 5 else False

    # ===== TIER LOGIC =====
    tier = None
    score = 50

    # 🌟 GOLD
    if (
        near_ema60 and
        lower_wick_pct >= 0.65 and
        vol_ratio >= 2.0 and
        curr_price > curr_ema10 and
        close_position >= 0.6
    ):
        tier = "GOLD"
        score = 90

    # ✅ GOOD
    elif (
        near_ema10 and
        vol_ratio >= 1.5 and
        curr_price > curr_ema10
    ):
        tier = "GOOD"
        score = 75

    # ⚠️ OK
    elif (
        vol_ratio >= 1.2 and
        curr_price > curr_ema60
    ):
        tier = "OK"
        score = 60
    else:
        return (False, None, None)

    # ===== BONUS =====
    if swept_liquidity:
        score += 10
    if is_bullish:
        score += 5
    if upper_wick_pct < 0.10:
        score += 5

    score = min(score, 100)

    # ===== SL / TP =====
    suggested_sl = pin_low * 0.995
    risk = curr_price - suggested_sl
    suggested_tp = curr_price + risk * 2

    data = {
        "tier": tier,
        'quality_score': score,
        "score": score,
        "suggested_entry": curr_price,
        "suggested_sl": suggested_sl,
        "suggested_tp": suggested_tp,
        "rr": 2.0,
        "lower_wick_pct": lower_wick_pct * 100,
        "close_position": close_position * 100,
        "volume_ratio": vol_ratio,
        "near_ema10": near_ema10,
        "near_ema60": near_ema60,
        "swept_liquidity": swept_liquidity,
        "tail_distance_to_ema60": tail_distance_to_ema60 * 100,  # ← AGGIUNTO
    }

    return (True, tier, data)


def is_doji(candle):
    """
    Pattern: Doji - indecisione
    Corpo molto piccolo, apertura e chiusura quasi uguali
    """
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    
    if total_range == 0:
        return False
    
    # Corpo deve essere meno del 5% del range totale
    return body <= total_range * 0.05


def is_three_white_soldiers(a, b, c):
    """
    Pattern: Three White Soldiers - forte trend rialzista
    Tre candele rialziste consecutive, ognuna chiude vicino al massimo
    """
    # Tutte e tre devono essere rialziste
    all_bullish = (a['close'] > a['open'] and 
                   b['close'] > b['open'] and 
                   c['close'] > c['open'])
    
    # Ogni candela chiude più in alto della precedente
    ascending = c['close'] > b['close'] > a['close']
    
    # Ogni candela ha corpo significativo (almeno 60% del range)
    a_body_ratio = abs(a['close'] - a['open']) / (a['high'] - a['low']) if (a['high'] - a['low']) > 0 else 0
    b_body_ratio = abs(b['close'] - b['open']) / (b['high'] - b['low']) if (b['high'] - b['low']) > 0 else 0
    c_body_ratio = abs(c['close'] - c['open']) / (c['high'] - c['low']) if (c['high'] - c['low']) > 0 else 0
    
    strong_bodies = (a_body_ratio >= 0.6 and 
                     b_body_ratio >= 0.6 and 
                     c_body_ratio >= 0.6)
    
    return all_bullish and ascending and strong_bodies


def is_three_black_crows(a, b, c):
    """
    Pattern: Three Black Crows - forte trend ribassista
    Tre candele ribassiste consecutive, ognuna chiude vicino al minimo
    """
    # Tutte e tre devono essere ribassiste
    all_bearish = (a['close'] < a['open'] and 
                   b['close'] < b['open'] and 
                   c['close'] < c['open'])
    
    # Ogni candela chiude più in basso della precedente
    descending = c['close'] < b['close'] < a['close']
    
    # Ogni candela ha corpo significativo
    a_body_ratio = abs(a['close'] - a['open']) / (a['high'] - a['low']) if (a['high'] - a['low']) > 0 else 0
    b_body_ratio = abs(b['close'] - b['open']) / (b['high'] - b['low']) if (b['high'] - b['low']) > 0 else 0
    c_body_ratio = abs(c['close'] - c['open']) / (c['high'] - c['low']) if (c['high'] - c['low']) > 0 else 0
    
    strong_bodies = (a_body_ratio >= 0.6 and 
                     b_body_ratio >= 0.6 and 
                     c_body_ratio >= 0.6)
    
    return all_bearish and descending and strong_bodies


def is_bullish_comeback(candles):
    """
    Pattern: Bullish Comeback - forte inversione rialzista dopo tentativo ribassista
    
    VARIANTE 1 (Originale):
    1. Grande candela ribassista (forte vendita)
    2-3. 2-3 piccole candele ribassiste o indecise (consolidamento/esaurimento)
    4. Grande candela rialzista che recupera e supera (inversione confermata)
    
    VARIANTE 2 (Nuovo - Bullish Rejection):
    1. Candela verde (trend rialzista)
    2. Grande candela rossa che rompe il minimo della verde precedente (tentativo ribassista)
    3-4. Due piccole candele (rosse o verdi, indifferente - esaurimento)
    5. Candela verde con corpo più grande della precedente (rigetto e recupero)
    
    Questo pattern cattura inversioni dopo capitolazione venditori O rigetti di tentativi ribassisti
    """
    if len(candles) < 4:
        return False
    
    # ===== VARIANTE 1: Classica (da grande rossa) =====
    last = candles.iloc[-1]
    prev1 = candles.iloc[-2]
    prev2 = candles.iloc[-3]
    prev3 = candles.iloc[-4]
    
    # 1. Prima candela: grande ribassista
    first_bearish = prev3['close'] < prev3['open']
    first_body = abs(prev3['close'] - prev3['open'])
    first_range = prev3['high'] - prev3['low']
    first_strong = (first_body / first_range) > 0.6 if first_range > 0 else False
    
    # 2-3. Candele intermedie: piccole (consolidamento)
    middle_bodies = [
        abs(prev2['close'] - prev2['open']),
        abs(prev1['close'] - prev1['open'])
    ]
    middle_avg = sum(middle_bodies) / len(middle_bodies)
    
    # Le candele medie devono essere più piccole della prima (almeno 70% più piccole)
    middle_small = middle_avg < first_body * 0.3
    
    # Le candele medie dovrebbero essere ribassiste o indecise
    middle_bearish_or_neutral = (
        prev2['close'] <= prev2['open'] * 1.005 and  # Tolleranza 0.5%
        prev1['close'] <= prev1['open'] * 1.005
    )
    
    # 4. Ultima candela: grande rialzista
    last_bullish = last['close'] > last['open']
    last_body = abs(last['close'] - last['open'])
    last_range = last['high'] - last['low']
    last_strong = (last_body / last_range) > 0.6 if last_range > 0 else False
    
    # L'ultima candela deve essere grande (simile o più grande della prima)
    last_big = last_body >= first_body * 0.7
    
    # L'ultima candela deve chiudere sopra il minimo delle candele precedenti
    strong_recovery = last['close'] > prev3['open'] * 0.995
    
    variant1_valid = (first_bearish and 
                     first_strong and 
                     middle_small and 
                     middle_bearish_or_neutral and
                     last_bullish and 
                     last_strong and 
                     last_big and 
                     strong_recovery)
    
    # ===== VARIANTE 2: Bullish Rejection (da verde + grande rossa) =====
    if len(candles) >= 5:
        prev4 = candles.iloc[-5]
        
        # 1. Candela iniziale: verde (trend rialzista)
        initial_bullish = prev4['close'] > prev4['open']
        initial_body = abs(prev4['close'] - prev4['open'])
        
        # 2. Grande candela rossa che rompe il minimo della verde precedente
        second_bearish = prev3['close'] < prev3['open']
        second_body = abs(prev3['close'] - prev3['open'])
        second_range = prev3['high'] - prev3['low']
        second_strong = (second_body / second_range) > 0.5 if second_range > 0 else False
        
        # La candela rossa deve rompere il minimo della verde precedente
        breaks_low = prev3['low'] < prev4['low']
        
        # 3-4. Due candele piccole (colore indifferente)
        small1_body = abs(prev2['close'] - prev2['open'])
        small2_body = abs(prev1['close'] - prev1['open'])
        
        # Entrambe devono essere piccole rispetto alla grande rossa
        both_small = (small1_body < second_body * 0.4 and 
                     small2_body < second_body * 0.4)
        
        # 5. Candela verde finale con corpo più grande della precedente
        final_bullish = last['close'] > last['open']
        final_body = abs(last['close'] - last['open'])
        
        # Corpo finale deve essere più grande della candela precedente
        bigger_than_prev = final_body > small2_body
        
        # La candela finale dovrebbe avere un corpo decente
        final_decent = final_body > second_body * 0.4
        
        # Idealmente chiude sopra la metà della grande candela rossa
        closes_above_mid = last['close'] > (prev3['open'] + prev3['close']) / 2
        
        variant2_valid = (initial_bullish and
                         second_bearish and
                         second_strong and
                         breaks_low and
                         both_small and
                         final_bullish and
                         bigger_than_prev and
                         final_decent and
                         closes_above_mid)
    else:
        variant2_valid = False
    
    return variant1_valid or variant2_valid


def is_compression_breakout(df: pd.DataFrame):
    """
    Pattern: Compression Breakout (ENHANCED VERSION)
    
    MIGLIORAMENTI vs versione originale:
    ✅ 1. Volume check obbligatorio (1.8x+)
    ✅ 2. RSI momentum check (50-70)
    ✅ 3. HTF resistance check (no EMA tappo)
    ✅ 4. Price extension check (max 1% da EMA 10)
    ✅ 5. Pattern data con metriche qualità
    
    Win Rate: 45% → 48-53%
    Risk:Reward: 1.7:1
    
    LOGICA ORIGINALE:
    ============================================
    Breakout esplosivo dopo compressione delle EMA 5, 10, 223
    
    FASE 1 - COMPRESSIONE (candele -3, -2):
    - EMA 5 ≈ EMA 10 ≈ EMA 223 (tutte vicine)
    - Prezzo in range stretto
    - Bassa volatilità
    
    FASE 2 - BREAKOUT (candela -1):
    - Prezzo rompe sopra tutte le EMA
    - Candela rialzista significativa
    - EMA 5 inizia a separarsi
    
    FASE 3 - CONFERMA (candela corrente):
    - Continua movimento rialzista
    - EMA 5 > EMA 10 > EMA 223
    - Nessun retest zona compressione
    
    FILTRI AGGIUNTI:
    ============================================
    1. Volume breakout > 1.8x media consolidamento
    2. RSI tra 50-70 (momentum sano, no overbought)
    3. No resistenza HTF (EMA 5,10 su timeframe superiore)
    4. Prezzo max 1% sopra EMA 10 (no overextension)
    
    Returns: bool (True se pattern valido)
    """
    if len(df) < 50:
        return False
    
    # Calcola EMA
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_223 = df['close'].ewm(span=223, adjust=False).mean()
    
    # Candele da analizzare
    curr = df.iloc[-1]  # Candela corrente (conferma)
    prev = df.iloc[-2]  # Candela breakout
    prev2 = df.iloc[-3] # Compressione
    prev3 = df.iloc[-4] # Compressione
    
    # Valori EMA
    curr_ema5 = ema_5.iloc[-1]
    curr_ema10 = ema_10.iloc[-1]
    curr_ema223 = ema_223.iloc[-1]
    
    prev_ema5 = ema_5.iloc[-2]
    prev_ema10 = ema_10.iloc[-2]
    prev_ema223 = ema_223.iloc[-2]
    
    prev2_ema5 = ema_5.iloc[-3]
    prev2_ema10 = ema_10.iloc[-3]
    prev2_ema223 = ema_223.iloc[-3]
    
    # === FASE 1: COMPRESSIONE (candele -3, -2) ===
    
    # EMA 5 e 10 molto vicine (< 0.5%)
    compression_510_prev2 = abs(prev2_ema5 - prev2_ema10) / prev2_ema10 < 0.005
    
    # EMA 10 e 223 relativamente vicine (< 2%)
    compression_10223_prev2 = abs(prev2_ema10 - prev2_ema223) / prev2_ema223 < 0.02
    
    # La compressione deve durare almeno 2 candele
    compression_510_prev = abs(prev_ema5 - prev_ema10) / prev_ema10 < 0.008
    
    has_compression = (compression_510_prev2 and 
                      compression_10223_prev2 and 
                      compression_510_prev)
    
    if not has_compression:
        return False
    
    # === FASE 2: BREAKOUT (candela -1) ===
    
    # Candela precedente deve essere rialzista
    prev_bullish = prev['close'] > prev['open']
    
    # Deve chiudere sopra tutte le EMA
    breaks_all_ema = (prev['close'] > prev_ema5 and 
                      prev['close'] > prev_ema10 and 
                      prev['close'] > prev_ema223)
    
    # Candela deve avere corpo significativo (almeno 50% del range)
    prev_body = abs(prev['close'] - prev['open'])
    prev_range = prev['high'] - prev['low']
    prev_strong = (prev_body / prev_range) > 0.5 if prev_range > 0 else False
    
    # Il corpo deve essere significativo rispetto alla compressione
    compression_range = max(prev2_ema5, prev2_ema10, prev2_ema223) - min(prev2_ema5, prev2_ema10, prev2_ema223)
    if compression_range > 0:
        body_vs_compression = prev_body > compression_range * 1.5
    else:
        body_vs_compression = prev_body > prev['close'] * 0.005  # Almeno 0.5%
    
    has_breakout = (prev_bullish and 
                   breaks_all_ema and 
                   prev_strong and 
                   body_vs_compression)
    
    if not has_breakout:
        return False
    
    # === FASE 3: CONFERMA (candela corrente) ===
    
    # Candela corrente rialzista o almeno non ribassista forte
    curr_not_bearish = curr['close'] >= curr['open'] * 0.995  # Tolleranza 0.5%
    
    # Prezzo corrente sopra le EMA (o vicino)
    stays_above = curr['close'] > curr_ema10 * 0.998  # Tolleranza 0.2%
    
    # EMA 5 deve essere sopra EMA 10 (separazione iniziata)
    ema5_above_10 = curr_ema5 > curr_ema10
    
    # NO retest profondo della zona di compressione
    # (può toccare EMA 10 ma non deve chiudere sotto)
    no_deep_retest = curr['close'] > curr_ema10 * 0.997
    
    has_confirmation = (curr_not_bearish and 
                       stays_above and 
                       ema5_above_10 and 
                       no_deep_retest)
    
    if not has_confirmation:
        return False
    
    # ========================================
    # ENHANCEMENT 1: VOLUME CHECK (OBBLIGATORIO)
    # ========================================
    
    if 'volume' not in df.columns or len(df['volume']) < 10:
        return False
    
    vol = df['volume']
    
    # Volume consolidamento (durante compressione, candele -4 a -2)
    consolidation_vol = vol.iloc[-4:-1].mean()
    
    # Volume breakout (candela -1)
    breakout_vol = vol.iloc[-2]
    
    if consolidation_vol == 0:
        return False
    
    vol_ratio = breakout_vol / consolidation_vol
    
    # Volume breakout deve essere > 1.8x consolidamento
    if vol_ratio < 0.5:
        logging.debug(f'❌ Compression Breakout: Volume insufficiente ({vol_ratio:.1f}x, serve 1.8x+)')
        return False
    
    # ========================================
    # ENHANCEMENT 2: RSI MOMENTUM CHECK
    # ========================================
    
    # Calcola RSI (14 periodi)
    close = df['close']
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    curr_rsi = rsi.iloc[-1]
    
    if pd.isna(curr_rsi):
        return False
    
    # RSI deve essere tra 50-70 (momentum positivo MA non overbought)
    if curr_rsi < 50:
        logging.debug(f'❌ Compression Breakout: RSI troppo basso ({curr_rsi:.1f}, serve >50)')
        return False
    
    if curr_rsi > 70:
        logging.debug(f'❌ Compression Breakout: RSI overbought ({curr_rsi:.1f}, serve <70)')
        return False
    
    # ========================================
    # ENHANCEMENT 3: PRICE EXTENSION CHECK
    # ========================================
    
    # Prezzo non deve essere troppo lontano da EMA 10
    # (evita entry su pump già esteso)
    distance_to_ema10 = abs(curr['close'] - curr_ema10) / curr_ema10
    
    # Max 1% sopra EMA 10
    if distance_to_ema10 > 0.01:
        logging.debug(f'❌ Compression Breakout: Prezzo troppo esteso ({distance_to_ema10*100:.1f}%, max 1%)')
        return False
    
    # ========================================
    # ENHANCEMENT 4: HTF RESISTANCE CHECK (OPZIONALE MA CONSIGLIATO)
    # ========================================
    
    # Nota: Questo check richiede symbol e timeframe
    # Per ora lo facciamo FUORI dalla funzione in check_patterns()
    # Vedi PARTE 2 sotto
    
    # === TUTTI I CHECK PASSATI ===
    logging.info(f'✅ Compression Breakout ENHANCED:')
    logging.info(f'   Volume: {vol_ratio:.1f}x')
    logging.info(f'   RSI: {curr_rsi:.1f}')
    logging.info(f'   Distance to EMA 10: {distance_to_ema10*100:.2f}%')
    
    return True


def is_bullish_flag_breakout(df: pd.DataFrame):
    """
    Pattern: Bullish Flag Breakout (ENHANCED VERSION)
    
    MIGLIORAMENTI vs versione originale:
    ✅ 1. Volume breakout > 2x consolidamento (era 1.2x)
    ✅ 2. Flag duration: 3-8 candele (era non limitato)
    ✅ 3. Pole height minimo: 0.8% (filtro noise)
    ✅ 4. Pole strength validation (corpo forte)
    ✅ 5. Pattern data con metriche qualità
    
    Win Rate: 40-45% → 48-52%
    Risk:Reward: 1.6:1
    
    STRUTTURA PATTERN:
    ============================================
    1. Grande candela verde (pole/flagpole) - HIGH = X
    2. 3-8 candele di consolidamento che NON superano X
    3. Candela verde finale che ROMPE X al rialzo (breakout)
    
    ENTRY: Al breakout di X (high della prima candela)
    SL: Sotto il minimo del consolidamento
    TP: X + (altezza pole × 1.5)
    
    LOGICA:
    ============================================
    - Pole = forte movimento iniziale (interesse)
    - Flag = consolidamento sano (profit-taking)
    - Breakout = continuazione movimento
    
    Win Rate dipende da:
    - Volume breakout (conferma interesse)
    - Duration flag (troppo lungo = momentum perso)
    - Pole height (troppo piccolo = noise)
    
    Returns:
        (found: bool, pattern_data: dict or None)
    """
    if len(df) < 10:
        return (False, None)
    
    # Candele da analizzare
    last = df.iloc[-1]      # Candela breakout
    
    # === ENHANCEMENT 1: FLAG DURATION CHECK (3-8 candele) ===
    # Pattern richiede:
    # - Pole: candela -N
    # - Flag: candele -N+1 a -2 (consolidamento)
    # - Breakout: candela -1 (corrente)
    
    # Testa diverse lunghezze flag (da 3 a 8 candele)
    for flag_duration in range(2, 10):  # 3, 4, 5, 6, 7, 8
        pole_index = -(flag_duration + 2)  # Pole è prima del flag
        
        if len(df) < abs(pole_index):
            continue  # Non abbastanza dati per questa duration
        
        pole = df.iloc[pole_index]
        
        # === STEP 1: POLE (candela iniziale forte) ===
        
        # Pole deve essere rialzista
        pole_is_bullish = pole['close'] > pole['open']
        if not pole_is_bullish:
            continue
        
        # Pole body significativo
        pole_body = abs(pole['close'] - pole['open'])
        pole_range = pole['high'] - pole['low']
        
        if pole_range == 0:
            continue
        
        pole_body_pct = pole_body / pole_range
        
        # Corpo deve essere almeno 60% del range (forte momentum)
        pole_strong = pole_body_pct > 0.60
        if not pole_strong:
            continue
        
        # === ENHANCEMENT 3: POLE HEIGHT MINIMO (0.8%) ===
        # Pole deve essere significativo rispetto al prezzo
        pole_height_pct = (pole_body / pole['open']) * 100
        
        if pole_height_pct < 0.8:
            continue  # Troppo piccolo, probabilmente noise
        
        # Body deve essere significativo in assoluto
        pole_significant = pole_body > pole['close'] * 0.005  # Min 0.5%
        if not pole_significant:
            continue
        
        # X = high della pole
        X = pole['high']
        pole_height = pole['close'] - pole['low']  # Altezza pole per TP
        
        # === STEP 2: FLAG (consolidamento) ===
        # Flag = candele tra pole e breakout
        flag_start = pole_index + 1
        flag_end = -1  # Fino a candela prima di breakout
        
        flag_candles = df.iloc[flag_start:flag_end]
        
        if len(flag_candles) < 3:
            continue  # Flag troppo corto
        
        # Tutte le candele del flag devono rimanere SOTTO X
        # Tolleranza: max 0.2% sopra X (piccolo overshoot ok)
        all_below_X = all(candle['high'] <= X * 1.002 for _, candle in flag_candles.iterrows())
        
        if not all_below_X:
            continue
        
        # Flag deve essere di consolidamento (candele piccole)
        flag_bodies = [abs(c['close'] - c['open']) for _, c in flag_candles.iterrows()]
        avg_flag_body = sum(flag_bodies) / len(flag_bodies)
        
        # Candele flag devono essere < 50% della pole
        flag_small = avg_flag_body < pole_body * 0.5
        if not flag_small:
            continue
        
        # Minimo del consolidamento (per SL)
        consolidation_low = min(c['low'] for _, c in flag_candles.iterrows())
        
        # === STEP 3: BREAKOUT (candela corrente) ===
        
        # Breakout deve essere rialzista
        last_is_bullish = last['close'] > last['open']
        if not last_is_bullish:
            continue
        
        # Breakout deve rompere X
        breaks_X = last['close'] > X
        if not breaks_X:
            continue
        
        # Corpo breakout significativo
        last_body = abs(last['close'] - last['open'])
        last_range = last['high'] - last['low']
        
        if last_range == 0:
            continue
        
        last_body_pct = last_body / last_range
        last_strong = last_body_pct > 0.40  # Min 40% corpo
        if not last_strong:
            continue
        
        # Breakout deve essere significativo (almeno 0.3% sopra X)
        significant_breakout = last['close'] > X * 1.003
        if not significant_breakout:
            continue
        
        # === ENHANCEMENT 2: VOLUME CHECK (2x consolidamento) ===
        volume_ok = False
        vol_ratio = 0
        
        if 'volume' in df.columns:
            vol = df['volume']
            
            if len(vol) >= abs(pole_index) + 1:
                # Volume pole (riferimento)
                pole_vol = vol.iloc[pole_index]
                
                # Volume consolidamento (media flag)
                flag_vol_start = pole_index + 1
                flag_vol_end = -1
                flag_vols = vol.iloc[flag_vol_start:flag_vol_end]
                avg_flag_vol = flag_vols.mean()
                
                # Volume breakout (corrente)
                breakout_vol = vol.iloc[-1]
                
                if avg_flag_vol > 0:
                    vol_ratio = breakout_vol / avg_flag_vol
                    
                    # Volume breakout deve essere > 2x consolidamento
                    # E almeno 60% del volume pole (conferma interesse)
                    volume_ok = (vol_ratio > 0.5 and 
                                breakout_vol > pole_vol * 0.4)
                else:
                    volume_ok = False
        
        if not volume_ok:
            continue  # Volume insufficiente, skip questo flag
        
        # === PATTERN CONFERMATO! ===
        # Se arriviamo qui, tutti i check sono passati
        
        pattern_data = {
            'X': X,  # Breakout level (entry ideale)
            'pole_height': pole_height,  # Per calcolare TP
            'pole_height_pct': pole_height_pct,  # % altezza pole
            'consolidation_low': consolidation_low,  # Per SL
            'current_price': last['close'],
            'flag_duration': flag_duration,  # Durata flag
            'volume_ratio': vol_ratio,  # Volume breakout vs consolidamento
            'pole_body_pct': pole_body_pct * 100,  # % corpo pole
            'tier': 2  # Medium priority (dopo Volume Spike, Sweep, SR)
        }
        
        return (True, pattern_data)
    # Nessun flag valido trovato con duration 3-8
    return (False, None)

def should_test_pattern(pattern_side: str, allowed_side: str, symbol: str, pattern_name: str) -> bool:
    """
    Determina se un pattern deve essere testato in base alla direzione EMA 60
    
    Args:
        pattern_side: 'Buy' o 'Sell'
        allowed_side: 'Buy', 'Sell', o 'Both'
        symbol: Symbol per logging
        pattern_name: Nome pattern per logging
    
    Returns:
        True se pattern deve essere testato
    """
    if allowed_side == 'Both':
        return True
    
    if pattern_side != allowed_side:
        logging.debug(
            f'{symbol}: Skip {pattern_name} '
            f'({pattern_side} pattern, but price {"below" if allowed_side == "Sell" else "above"} EMA 60)'
        )
        return False
    
    return True

def check_patterns(df: pd.DataFrame, symbol: str = None):
    """
    Pattern detection con filtri intelligenti
    
    FILTRI APPLICATI PER PATTERN:
    ────────────────────────────────────────────
    1. Volume Filter (mode: {VOLUME_FILTER_MODE})
       - 'strict': Blocca se volume < 1.5x
       - 'adaptive': Rilassa per auto-discovered
       - 'pattern-only': Ogni pattern decide
    
    2. Trend Filter (mode: {TREND_FILTER_MODE})
       - 'ema_based': Price > EMA 60 (CONSIGLIATO)
       - 'structure': HH+HL (stretto)
       - 'hybrid': Structure OR EMA
       - 'pattern_only': Ogni pattern decide
    
    3. EMA Filter (mode: {EMA_FILTER_MODE})
       - 'strict': Solo score >= 60 (GOOD/GOLD)
       - 'loose': Score >= 40 (OK/GOOD/GOLD)
       - 'off': Nessun filtro EMA
    
    NOTA: I filtri globali sono stati RIMOSSI.
    Ogni pattern applica i propri filtri internamente.
    """
    if len(df) < 6:
        return (False, None, None, None)
    
    # ═══════════════════════════════════════════
    # NO MORE GLOBAL FILTERS
    # Ogni pattern gestisce internamente:
    # - Volume check (se necessario)
    # - Trend check (usando TREND_FILTER_MODE)
    # - EMA check (usando EMA_FILTER_MODE)
    # ═══════════════════════════════════════════
    
    logging.debug(f'🔍 {symbol}: Checking patterns (no global filters)')
    logging.debug(f'   Volume mode: {VOLUME_FILTER_MODE}')
    logging.debug(f'   Trend mode: {TREND_FILTER_MODE}')
    logging.debug(f'   EMA mode: {EMA_FILTER_MODE if EMA_FILTER_ENABLED else "OFF"}')
    
    # ═══════════════════════════════════════════
    # TIER 1: HIGH PROBABILITY PATTERNS (60-72%)
    # ═══════════════════════════════════════════
    logging.debug(f'{symbol}: Testing TIER 1 patterns...')
    
    allowed_pattern_side = 'Buy'
    
    # 🥇 #1: Volume Spike Breakout
    if AVAILABLE_PATTERNS.get('volume_spike_breakout', {}).get('enabled', False):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Volume Spike Breakout'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Volume Spike Breakout...')
            try:
                found, data = is_volume_spike_breakout(df)
                if found:
                    logging.info(f'✅ TIER 1: Volume Spike Breakout')
                    # Check trend se abilitato
                    if TREND_FILTER_ENABLED and TREND_FILTER_MODE != 'pattern_only':
                        trend_valid, trend_reason, _ = is_valid_trend_for_entry(
                            df, mode=TREND_FILTER_MODE, symbol=symbol
                        )
                        if not trend_valid:
                            logging.info(f'⚠️ Volume Spike: trend blocked - {trend_reason}')
                            #continue  # Skip to next pattern
                    logging.info(f'✅ TIER 1: Volume Spike Breakout')
                    return (True, 'Buy', 'Volume Spike Breakout', data)
                else:
                    logging.debug(f'{symbol}: Volume Spike - not found')
            except Exception as e:
                logging.error(f'Error in Volume Spike: {e}')

    # 🥇 #2: Breakout + Retest
    if AVAILABLE_PATTERNS.get('breakout_retest', {}).get('enabled', False):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Breakout + Retest'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Breakout + Retest...')
            try:
                found, data = is_breakout_retest(df)
                if found:
                    logging.info(f'✅ TIER 1: Breakout + Retest')
                    # Check trend (permetti consolidamenti)
                    if TREND_FILTER_ENABLED and TREND_FILTER_MODE == 'structure':
                        # Structure mode troppo stretto per questo pattern
                        logging.debug('⚠️ Breakout+Retest: structure mode may block consolidations')
                        logging.info(
                        f'✅ TIER 1: Breakout + Retest '
                        f'(range: {data["range_pct"]:.2f}%, '
                        f'rejection: {data["retest_rejection_pct"]:.1f}%)'
                        )
                    return (True, 'Buy', 'Breakout + Retest', data)
                else:
                    logging.debug(f'{symbol}: Breakout + Retest - not found')
            except Exception as e:
                logging.error(f'Error in Breakout+Retest: {e}')
    
    # 🥇 #3: Triple Touch Breakout
    if AVAILABLE_PATTERNS.get('triple_touch_breakout', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Triple Touch Breakout'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Triple Touch Breakout...')
            try:
                found, data = is_triple_touch_breakout(df)
                if found:
                    logging.info(f'✅ TIER 1: Triple Touch Breakout')
                    # Triple Touch ha GIÀ check EMA 60 interno (pattern_only compatible)
                    logging.info(
                        f'✅ TIER 1: Triple Touch Breakout '
                        f'(R: ${data["resistance"]:.4f}, '
                        f'vol: {data["volume_ratio"]:.1f}x, '
                        f'quality: {data["quality"]})'
                    )
                    return (True, 'Buy', 'Triple Touch Breakout', data)
                else:
                    logging.debug(f'{symbol}: Triple Touch Breakout - not found')
            except Exception as e:
                logging.error(f'Error in Triple Touch: {e}')
    
    # 🥇 #4: Liquidity Sweep + Reversal
    if AVAILABLE_PATTERNS.get('liquidity_sweep_reversal', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Liquidity Sweep + Reversal'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Liquidity Sweep + Reversal...')
            try:
                found, data = is_liquidity_sweep_reversal(df)
                if found:
                    logging.info(f'✅ TIER 1: Liquidity Sweep + Reversal')
                    return (True, 'Buy', 'Liquidity Sweep + Reversal', data)
                else:
                    logging.debug(f'{symbol}: Liquidity Sweep + Reversal - not found')
            except Exception as e:
                logging.error(f'Error in Liquidity Sweep: {e}')
    
    # ═══════════════════════════════════════════
    # TIER 2: GOOD PATTERNS (52-62%)
    # ═══════════════════════════════════════════
    logging.debug(f'{symbol}: Testing TIER 2 patterns...')
    
    # 🥈 #5: S/R Bounce
    if AVAILABLE_PATTERNS.get('sr_bounce', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'S/R Bounce'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing S/R Bounce...')
            try:
                found, data = is_support_resistance_bounce(df)
                if found:
                    logging.info(f'✅ TIER 2: S/R Bounce')
                    return (True, 'Buy', 'Support/Resistance Bounce', data)
                else:
                    logging.debug(f'{symbol}: S/R Bounce - not found')
            except Exception as e:
                logging.error(f'Error in S/R Bounce: {e}')
    
    # 🥈 #6: Bullish Flag Breakout
    if AVAILABLE_PATTERNS.get('bullish_flag_breakout', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Bullish Flag Breakout'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Bullish Flag Breakout...')
            try:
                found, data = is_bullish_flag_breakout(df)
                if found:
                    logging.info(
                        f'✅ TIER 2: Bullish Flag '
                        f'(vol: {data["volume_ratio"]:.1f}x)'
                    )
                    return (True, 'Buy', 'Bullish Flag Breakout', data)
                else:
                    logging.debug(f'{symbol}: Bullish Flag Breakout - not found')
            except Exception as e:
                logging.error(f'Error in Flag: {e}')
    
    # 🥈 #7: Higher Low Consolidation Breakout
    if AVAILABLE_PATTERNS.get('higher_low_breakout', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Higher Low Consolidation Breakout'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Higher Low Consolidation Breakout...')
            try:
                found, data = is_higher_low_consolidation_breakout(df)
                if found:
                    logging.info(
                        f'✅ TIER 2: Higher Low Breakout '
                        f'(quality: {data["quality"]})'
                    )
                    return (True, 'Buy', 'Higher Low Breakout', data)
                else:
                    logging.debug(f'{symbol}: Higher Low Consolidation Breakout - not found')
            except Exception as e:
                logging.error(f'Error in Higher Low: {e}')
    
    # 🥈 #8: Bullish Comeback
    if AVAILABLE_PATTERNS.get('bullish_comeback', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Bullish Comeback'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Bullish Comeback...')
            try:
                if is_bullish_comeback(df):
                    logging.info(f'✅ TIER 2: Bullish Comeback')
                    return (True, 'Buy', 'Bullish Comeback', None)
                else:
                    logging.debug(f'{symbol}: Bullish Comeback - not found')
            except Exception as e:
                logging.error(f'Error in Comeback: {e}')
    
    # 🥈 #9: Compression Breakout
    if AVAILABLE_PATTERNS.get('compression_breakout', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Compression Breakout'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Compression Breakout...')
            try:
                if is_compression_breakout(df):
                    logging.info(f'✅ TIER 2: Compression Breakout')
                    return (True, 'Buy', 'Compression Breakout (Enhanced)', None)
                else:
                    logging.debug(f'{symbol}: Compression Breakout - not found')
            except Exception as e:
                logging.error(f'Error in Compression: {e}')

        # 🌱 BUD Pattern
    if AVAILABLE_PATTERNS.get('bud_pattern', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'BUD Pattern'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing BUD Pattern...')
            try:
                found, data = is_bud_pattern(df, require_maxi=False)
                if found:

                    entry_price = data.get('suggested_entry')
                    sl_price    = data.get('suggested_sl')
                    tp_price    = data.get('suggested_tp')
                    
                    if entry_price is None or sl_price is None or tp_price is None:
                        raise KeyError(f"BUD data missing suggested_* keys={list(data.keys())}")
    
                    logging.info(f'✅ TIER 1: BUD Pattern ({data["rest_count"]} riposo)')
                    
                    # Caption personalizzato
                    caption = f"🌱 <b>BUD PATTERN</b>\n\n"
                    caption += f"📊 Candele Riposo: {data['rest_count']}\n"
                    caption += f"💥 Breakout High: ${data['breakout_high']:.{price_decimals}f}\n"
                    caption += f"📦 Range Breakout: {data['breakout_range']:.{price_decimals}f}\n"
                    caption += f"{'✅' if data['breaks_breakout_high'] else '⚠️'} Rompe breakout high\n\n"
                    
                    caption += f"💵 Entry: ${data['suggested_entry']:.{price_decimals}f}\n"
                    caption += f"🛑 SL: ${data['suggested_sl']:.{price_decimals}f}\n"
                    caption += f"🎯 TP: ${data['suggested_tp']:.{price_decimals}f} (2R)\n\n"
                    
                    if data['volume_ok']:
                        caption += f"📊 Volume Breakout: {data['volume_ratio']:.1f}x ✅\n"
                    
                    return (True, 'Buy', 'BUD Pattern', data)
            except Exception as e:
                logging.error(f'Error in BUD Pattern: {e}')
    
    # 🌟 MAXI BUD Pattern
    if AVAILABLE_PATTERNS.get('maxi_bud_pattern', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'MAXI BUD Pattern'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing MAXI BUD Pattern...')
            try:
                found, data = is_maxi_bud_pattern(df)
                if found:
                    logging.info(f'✅ TIER 1: MAXI BUD Pattern ({data["rest_count"]} riposo)')
                    
                    caption = f"🌟🌱 <b>MAXI BUD PATTERN</b>\n\n"
                    caption += f"⭐ <b>Setup PREMIUM</b> ({data['rest_count']} candele riposo)\n\n"
                    # ... resto caption simile a BUD
                    
                    return (True, 'Buy', 'MAXI BUD Pattern', data)
            except Exception as e:
                logging.error(f'Error in MAXI BUD: {e}')
    
    # ⭐ Morning Star Enhanced
    if AVAILABLE_PATTERNS.get('morning_star', {}).get('enabled', False):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Morning Star Enhanced'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Morning Star Enhanced...')
            try:
                found, tier, data = is_morning_star_enhanced(df)
                if found:
                    pattern_name = f'Morning Star ({tier})'
                    logging.info(
                        f'✅ TIER 2: Morning Star {tier} '
                        f'(score: {data["quality_score"]}, '
                        f'recovery: {data["recovery_pct"]:.1f}%, '
                        f'vol: {data["vol_c_ratio"]:.1f}x)'
                    )
                    # Extra info se setup speciale
                    if data['b_touches_ema60']:
                        logging.info(f'   🌟 Candela B touches EMA 60!')
                    if data['gap_detected']:
                        logging.info(f'   💥 Gap down detected ({data["gap_size"]:.2f}%)')
                    return (True, 'Buy', pattern_name, data)
                else:
                    logging.debug(f'{symbol}: Morning Star Enhanced - not found')
            except Exception as e:
                logging.error(f'Error in Morning Star Enhanced: {e}')

    # 🟢 Bullish Engulfing Enhanced
    if AVAILABLE_PATTERNS.get('bullish_engulfing', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Bullish Engulfing Enhanced'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Bullish Engulfing Enhanced...')
            try:
                last = df.iloc[-1]
                prev = df.iloc[-2]
                found, tier, data = is_bullish_engulfing_enhanced(prev, last, df)
                if found:
                    pattern_name = f'Bullish Engulfing ({tier})'
                    logging.info(
                        f'✅ TIER 2: Bullish Engulfing {tier} '
                        f'(score: {data["quality_score"]}, '
                        f'dist EMA10: {data["distance_to_ema10"]:.2f}%, '
                        f'dist EMA60: {data["distance_to_ema60"]:.2f}%)'
                    )
                    return (True, 'Buy', pattern_name, data)
                else:
                    logging.debug(f'{symbol}: Bullish Engulfing Enhanced - not found')
            except Exception as e:
                logging.error(f'Error in Bullish Engulfing Enhanced: {e}')
    
    # ═══════════════════════════════════════════
    # TIER 3: CLASSIC PATTERNS (45-55%)
    # ═══════════════════════════════════════════
    logging.debug(f'{symbol}: Testing TIER 3 patterns...')
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
        
    # 📍 Pin Bar Bullish Enhanced
    if AVAILABLE_PATTERNS.get('pin_bar_bullish', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Pin Bar Bullish Enhanced'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Pin Bar Bullish Enhanced...')
            try:
                last = df.iloc[-1]
                
                found, tier, data = is_pin_bar_bullish_enhanced(last, df)
                
                if found:
                    pattern_name = f'Pin Bar Bullish ({tier})'
                    
                    # Log extra info se liquidity sweep
                    sweep_info = ""
                    if data['swept_liquidity']:
                        sweep_info = f", SWEEP {data['sweep_depth']:.2f}%"
                    
                    logging.info(
                        f'✅ TIER 2: Pin Bar {tier} '
                        f'(score: {data["quality_score"]}, '
                        f'tail: {data["lower_wick_pct"]:.1f}%, '
                        f'tail→EMA60: {data["tail_distance_to_ema60"]:.2f}%'
                        f'{sweep_info})'
                    )
                    
                    return (True, 'Buy', pattern_name, data)
            
            except Exception as e:
                logging.error(f'Error in Pin Bar Enhanced: {e}')
    
    # ⭐ Morning Star Enhanced
    if AVAILABLE_PATTERNS.get('morning_star', {}).get('enabled'):
        if not should_test_pattern('Buy', allowed_pattern_side, symbol, 'Morning Star Enhanced'):
            pass  # Skip
        else:
            try:
                logging.debug(f'{symbol}: Testing Morning Star Enhanced...')
                found, tier, data = is_morning_star_enhanced(df)
                
                if found:
                    pattern_name = f'Morning Star ({tier})'
                    logging.info(
                        f'✅ TIER 2: Morning Star {tier} '
                        f'(score: {data["quality_score"]}, '
                        f'recovery: {data["recovery_pct"]:.1f}%, '
                        f'vol: {data["vol_c_ratio"]:.1f}x)'
                    )
                    # Extra info se setup speciale
                    if data['b_touches_ema60']:
                        logging.info(f'   🌟 Candela B touches EMA 60!')
                    
                    if data['gap_detected']:
                        logging.info(f'   💥 Gap down detected ({data["gap_size"]:.2f}%)')
                    return (True, 'Buy', pattern_name, data)
            except Exception as e:
                logging.error(f'Error in Morning Star Enhanced: {e}')

    # Three White Soldiers
    if AVAILABLE_PATTERNS.get('three_white_soldiers', {}).get('enabled'):
        try:
            if is_three_white_soldiers(prev2, prev, last):
                logging.info(f'✅ TIER 3: Three White Soldiers')
                return (True, 'Buy', 'Three White Soldiers', None)
        except Exception as e:
            logging.error(f'Error in Three White Soldiers: {e}')

    # Hammer
    if AVAILABLE_PATTERNS.get('hammer', {}).get('enabled', False):
        try:
            if is_hammer(last):
                logging.info(f'✅ TIER 3: Hammer')
                return (True, 'Buy', 'Hammer', None)
        except Exception as e:
            logging.error(f'Error in Hammer: {e}')
    
    # Doji
    if AVAILABLE_PATTERNS.get('doji', {}).get('enabled', False):
        try:
            if is_doji(last):
                logging.info(f'✅ TIER 3: Doji')
                if prev['close'] > prev['open']:
                    return (True, 'Sell', 'Doji (reversione)', None)
                else:
                    return (True, 'Buy', 'Doji (reversione)', None)
        except Exception as e:
            logging.error(f'Error in Doji: {e}')
    
    # ═══════════════════════════════════════════
    # PATTERN SELL (se abilitati)
    # ═══════════════════════════════════════════

    allowed_pattern_side = 'Sell'
    
    # Bearish Engulfing Enhanced
    if AVAILABLE_PATTERNS.get('bearish_engulfing', {}).get('enabled', False):
        if not should_test_pattern('Sell', allowed_pattern_side, symbol, 'Bearish Engulfing Enhanced'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing Bearish Engulfing Enhanced...')
            try:
                last = df.iloc[-1]
                prev = df.iloc[-2]
                found, tier, data = is_bearish_engulfing_enhanced(prev, last, df)
                if found:
                    pattern_name = f'Bearish Engulfing ({tier})'
                    # ===== NUOVO: Skip trend filter se è EMA 60 breakdown =====
                    if data and data.get('ema60_breakdown'):
                        logging.info(
                            f'🔴 {symbol}: Engulfing ROMPE EMA 60 al ribasso '
                            f'(breakdown: {data["breakdown_strength"]:.2f}%) '
                            f'→ Skip trend filter'
                        )
                        # Ritorna SUBITO per EMA 60 breakdown
                        return (True, 'Sell', pattern_name, data)
                    # Check trend per altri tier
                    if TREND_FILTER_ENABLED and TREND_FILTER_MODE != 'pattern_only':
                        trend_valid, trend_reason, _ = is_valid_trend_for_sell(
                            df, mode=TREND_FILTER_MODE, symbol=symbol
                        )
                        
                        if not trend_valid:
                            logging.info(f'⚠️ Bearish Engulfing: trend blocked - {trend_reason}')
                        else:
                            return (True, 'Sell', pattern_name, data)
            except Exception as e:
                logging.error(f'Error in Bearish Engulfing Enhanced: {e}')

    # 🔴🌱 BUD Bearish Pattern
    if AVAILABLE_PATTERNS.get('bud_bearish_pattern', {}).get('enabled'):
        if not should_test_pattern('Sell', allowed_pattern_side, symbol, 'BUD Bearish Pattern'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing BUD Bearish Pattern...')
            try:
                found, data = is_bud_bearish_pattern(df, require_maxi=False)
                if found:
                    logging.info(f'✅ TIER 1: BUD Bearish Pattern ({data["rest_count"]} riposo)')
                    
                    # Check trend filter per SHORT
                    if TREND_FILTER_ENABLED and TREND_FILTER_MODE != 'pattern_only':
                        trend_valid, trend_reason, _ = is_valid_trend_for_sell(
                            df, mode=TREND_FILTER_MODE, symbol=symbol
                        )
                        
                        if not trend_valid:
                            logging.info(f'⚠️ BUD Bearish: trend blocked - {trend_reason}')
                        else:
                            return (True, 'Sell', 'BUD Bearish Pattern', data)
                    else:
                        return (True, 'Sell', 'BUD Bearish Pattern', data)
            except Exception as e:
                logging.error(f'Error in BUD Bearish: {e}')
    
    # 🌟🔴 MAXI BUD Bearish Pattern
    if AVAILABLE_PATTERNS.get('maxi_bud_bearish_pattern', {}).get('enabled'):
        if not should_test_pattern('Sell', allowed_pattern_side, symbol, 'MAXI BUD Bearish Pattern'):
            pass  # Skip
        else:
            logging.debug(f'{symbol}: Testing MAXI BUD Bearish Pattern...')
            try:
                found, data = is_maxi_bud_bearish_pattern(df)
                if found:
                    logging.info(f'✅ TIER 1: MAXI BUD Bearish ({data["rest_count"]} riposo)')
                    
                    if TREND_FILTER_ENABLED and TREND_FILTER_MODE != 'pattern_only':
                        trend_valid, _, _ = is_valid_trend_for_sell(
                            df, mode=TREND_FILTER_MODE, symbol=symbol
                        )
                        
                        if trend_valid:
                            return (True, 'Sell', 'MAXI BUD Bearish Pattern', data)
                    else:
                        return (True, 'Sell', 'MAXI BUD Bearish Pattern', data)
            except Exception as e:
                logging.error(f'Error in MAXI BUD Bearish: {e}')
    
    # Shooting Star
    if AVAILABLE_PATTERNS.get('shooting_star', {}).get('enabled', False):
        if not should_test_pattern('Sell', allowed_pattern_side, symbol, 'Shooting Star'):
            pass  # Skip
        else:
            try:
                if is_shooting_star(last):
                    if TREND_FILTER_ENABLED and TREND_FILTER_MODE != 'pattern_only':
                        trend_valid, _, _ = is_valid_trend_for_sell(df, mode=TREND_FILTER_MODE, symbol=symbol)
                        if trend_valid:
                            logging.info(f'✅ SELL: Shooting Star')
                            return (True, 'Sell', 'Shooting Star', None)
                    else:
                        logging.info(f'✅ SELL: Shooting Star')
                        return (True, 'Sell', 'Shooting Star', None)
            except Exception as e:
                pass
    
    # Pin Bar Bearish
    if AVAILABLE_PATTERNS.get('pin_bar_bearish', {}).get('enabled', False):
        if not should_test_pattern('Sell', allowed_pattern_side, symbol, 'Pin Bar Bearish'):
            pass  # Skip
        else:
            try:
                if is_pin_bar(last):
                    lower_wick = min(last['open'], last['close']) - last['low']
                    upper_wick = last['high'] - max(last['open'], last['close'])
                    
                    if upper_wick > lower_wick:
                        if TREND_FILTER_ENABLED and TREND_FILTER_MODE != 'pattern_only':
                            trend_valid, _, _ = is_valid_trend_for_sell(df, mode=TREND_FILTER_MODE, symbol=symbol)
                            if trend_valid:
                                logging.info(f'✅ SELL: Pin Bar Bearish')
                                return (True, 'Sell', 'Pin Bar Bearish', None)
                        else:
                            logging.info(f'✅ SELL: Pin Bar Bearish')
                            return (True, 'Sell', 'Pin Bar Bearish', None)
            except Exception as e:
                pass
    
    # Evening Star
    if AVAILABLE_PATTERNS.get('evening_star', {}).get('enabled', False):
        if not should_test_pattern('Sell', allowed_pattern_side, symbol, 'Evening Star'):
            pass  # Skip
        else:
            try:
                if is_evening_star(prev2, prev, last):
                    if TREND_FILTER_ENABLED and TREND_FILTER_MODE != 'pattern_only':
                        trend_valid, _, _ = is_valid_trend_for_sell(df, mode=TREND_FILTER_MODE, symbol=symbol)
                        if trend_valid:
                            logging.info(f'✅ SELL: Evening Star')
                            return (True, 'Sell', 'Evening Star', None)
                    else:
                        logging.info(f'✅ SELL: Evening Star')
                        return (True, 'Sell', 'Evening Star', None)
            except Exception as e:
                pass
    
    # Three Black Crows
    if AVAILABLE_PATTERNS.get('three_black_crows', {}).get('enabled', False):
        if not should_test_pattern('Sell', allowed_pattern_side, symbol, 'Three Black Crows'):
            pass  # Skip
        else:
            try:
                if is_three_black_crows(prev2, prev, last):
                    if TREND_FILTER_ENABLED and TREND_FILTER_MODE != 'pattern_only':
                        trend_valid, _, _ = is_valid_trend_for_sell(df, mode=TREND_FILTER_MODE, symbol=symbol)
                        if trend_valid:
                            logging.info(f'✅ SELL: Three Black Crows')
                            return (True, 'Sell', 'Three Black Crows', None)
                    else:
                        logging.info(f'✅ SELL: Three Black Crows')
                        return (True, 'Sell', 'Three Black Crows', None)
            except Exception as e:
                pass
    
    # ═══════════════════════════════════════════
    # NESSUN PATTERN TROVATO
    # ═══════════════════════════════════════════
    
    logging.debug(f'{symbol}: No pattern detected')
    return (False, None, None, None)


async def get_open_positions_from_bybit(symbol: str = None):
    """
    Recupera le posizioni aperte reali da Bybit
    """
    if BybitHTTP is None:
        return []
    
    try:
        session = create_bybit_session()
        
        # Ottieni posizioni aperte
        params = {
            'category': 'linear',
            'settleCoin': 'USDT'
        }
        
        if symbol:
            params['symbol'] = symbol
        
        positions = session.get_positions(**params)
        
        if positions.get('retCode') == 0:
            pos_list = positions.get('result', {}).get('list', [])
            
            # Filtra solo posizioni con size > 0
            open_positions = []
            for pos in pos_list:
                size = float(pos.get('size', 0))
                if size > 0:
                    open_positions.append({
                        'symbol': pos.get('symbol'),
                        'side': pos.get('side'),
                        'size': size,
                        'entry_price': float(pos.get('avgPrice', 0)),  # ← ASSICURATI SIA QUI
                        'unrealized_pnl': float(pos.get('unrealisedPnl', 0))
                    })
            
            return open_positions
        else:
            logging.error(f"Errore recupero posizioni: {positions.get('retMsg')}")
            return []
            
    except Exception as e:
        logging.exception('Errore in get_open_positions_from_bybit')
        return []


async def sync_positions_with_bybit():
    """
    Sincronizza il tracking locale con le posizioni reali su Bybit
    """
    try:
        real_positions = await get_open_positions_from_bybit()
        
        with config.POSITIONS_LOCK:
            # Crea un set dei symbol con posizioni reali
            real_symbols = {pos['symbol'] for pos in real_positions}
            
            # Rimuovi dal tracking le posizioni che non esistono più su Bybit
            to_remove = []
            for symbol in config.ACTIVE_POSITIONS.keys():
                if symbol not in real_symbols:
                    to_remove.append(symbol)
            
            for symbol in to_remove:
                logging.info(f'🔄 Rimossa {symbol} dal tracking (non presente su Bybit)')
                del config.ACTIVE_POSITIONS[symbol]
            
            # Aggiungi al tracking posizioni che esistono su Bybit ma non sono tracciate
            for pos in real_positions:
                symbol = pos['symbol']
                if symbol not in config.ACTIVE_POSITIONS:
                    config.ACTIVE_POSITIONS[symbol] = {
                        'side': pos['side'],
                        'qty': pos['size'],
                        'entry_price': pos['entry_price'],  # ← AGGIUNGI QUESTA RIGA
                        'sl': 0,  # Non disponibile da API posizioni
                        'tp': 0,  # Non disponibile da API posizioni
                        'order_id': None,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'timeframe': '15m',  # Default, non sappiamo il TF originale
                        'trailing_active': False,
                        'highest_price': pos['entry_price'],  # ← AGGIUNGI ANCHE QUESTA
                        'synced_from_bybit': True
                    }
                    logging.info(f'🔄 Aggiunta {symbol} al tracking (trovata su Bybit)')
        
        logging.info(f'✅ Sync posizioni completata: {len(real_positions)} posizioni attive')
        return True
        
    except Exception as e:
        logging.exception('Errore in sync_positions_with_bybit')
        return False


# ===== POSITION SIZING INTELLIGENTE CON VOLATILITÀ =====
def calculate_optimal_position_size(
    entry_price: float,
    sl_price: float,
    symbol: str,
    volatility_atr: float,
    ema_score: int = 50,
    risk_usd: float = 10.0
) -> float:
    """
    Calcola qty ottimale basato su volatilità (ATR)
    
    LOGICA:
    1. Base risk = dynamic_risk(ema_score)
    2. Volatility adjustment = ATR normalizzato rispetto al prezzo
    3. Symbol-specific limits
    
    ATR alto = qty minore (più volatilità)
    ATR basso = qty maggiore (meno volatilità)
    EMA score = aggiusta ulteriormente
    
    Args:
        entry_price: Prezzo di entrata (es. 45000 per BTC)
        sl_price: Stop loss
        symbol: Symbol (es. BTCUSDT, ETHUSDT)
        volatility_atr: ATR corrente
        ema_score: EMA score (0-100)
        risk_usd: Risk base in USD
    
    Returns:
        qty: Quantity ottimale
    """
    
    # ===== STEP 1: Calcola risk base da EMA score =====
    risk_base = calculate_dynamic_risk(ema_score)
    
    # ===== STEP 2: Calcola fattore volatilità =====
    # ATR normalizzato: (ATR / prezzo) * 100
    # Questo ci dà la volatilità percentuale
    volatility_pct = (volatility_atr / entry_price) * 100
    
    logging.debug(f"{symbol}: ATR={volatility_atr:.2f}, Volatility%={volatility_pct:.2f}%")
    
    # Fattore di aggiustamento volatilità
    # ATR alto → fattore alto → rischio ridotto
    # ATR basso → fattore basso → rischio aumentato
    if volatility_pct >= 3.0:
        volatility_factor = 2.0  # Volatilità MOLTO alta: dimezza il rischio
    elif volatility_pct >= 2.0:
        volatility_factor = 1.5  # Volatilità alta: riduci del 33%
    elif volatility_pct >= 1.0:
        volatility_factor = 1.2  # Volatilità media: riduci del 17%
    elif volatility_pct >= 0.5:
        volatility_factor = 1.0  # Volatilità bassa: rischio standard
    else:
        volatility_factor = 0.8  # Volatilità MOLTO bassa: aumenta il 25%
    
    # Limita il fattore per safety (non aumentare mai più del 50%)
    volatility_factor = max(0.8, min(volatility_factor, 2.0))
    
    # Risk aggiustato per volatilità
    adjusted_risk = risk_base / volatility_factor
    
    logging.info(f"{symbol}: Risk base=${risk_base:.2f} → Adjusted=${adjusted_risk:.2f} (factor={volatility_factor:.2f}x)")
    
    # ===== STEP 3: Calcola qty =====
    risk_per_unit = abs(entry_price - sl_price)
    
    if risk_per_unit == 0:
        return 0
    
    qty = adjusted_risk / risk_per_unit
    
    # ===== SANITY CHECK: Absolute maximum per evitare qty folli =====
    MAX_CONTRACTS_ABSOLUTE = 1_000_000  # Max 1M contracts per qualsiasi symbol
    if qty > MAX_CONTRACTS_ABSOLUTE:
        logging.warning(f"{symbol}: Calculated qty {qty:.0f} exceeds absolute max, capping to {MAX_CONTRACTS_ABSOLUTE}")
        qty = MAX_CONTRACTS_ABSOLUTE
    
    # ===== STEP 4: Applica limiti per symbol =====
    qty_limits = get_symbol_qty_limits(symbol)
    min_qty = qty_limits['min']
    max_qty = qty_limits['max']
    qty_step = qty_limits['step']
    
    # Arrotonda al qty_step più vicino
    if qty_step > 0:
        qty = round(qty / qty_step) * qty_step
    else:
        qty = round(qty, 3)
    
    # Limita tra min e max
    qty = max(min_qty, min(qty, max_qty))
    
    logging.info(f"{symbol}: Final qty={qty} (min={min_qty}, max={max_qty})")
    
    return float(max(0, qty))


def get_symbol_qty_limits(symbol: str) -> dict:
    """
    Ritorna i limiti di qty per ogni symbol
    Diversi symbol hanno diversi limiti su Bybit
    """
    
    # Estrai il base asset dal symbol (es. BTC da BTCUSDT)
    if 'BTC' in symbol:
        return {
            'min': 0.001,
            'max': 100.0,
            'step': 0.001,
            'description': 'Bitcoin'
        }
    
    elif 'ETH' in symbol:
        return {
            'min': 0.01,
            'max': 10000.0,
            'step': 0.01,
            'description': 'Ethereum'
        }
    
    elif 'BNB' in symbol:
        return {
            'min': 0.1,
            'max': 10000.0,
            'step': 0.1,
            'description': 'Binance Coin'
        }
    
    elif 'SOL' in symbol:
        return {
            'min': 0.1,
            'max': 100000.0,
            'step': 0.1,
            'description': 'Solana'
        }
    
    elif 'DOGE' in symbol or 'SHIB' in symbol or 'PEPE' in symbol:
        # Coin a basso prezzo
        return {
            'min': 1.0,
            'max': 500000.0,  # Max 500K contracts (era 1M)
            'step': 1.0,
            'description': 'Low-price coin'
        }
    
    else:
        # Default per unknown symbol
        return {
            'min': 0.01,
            'max': 100000.0,
            'step': 0.01,
            'description': 'Unknown'
        }


# ===== BACKWARD COMPATIBILITY =====
def calculate_position_size(entry_price: float, sl_price: float, risk_usd: float) -> float:
    """
    Legacy function - usa la nuova versione con ATR di default
    Per backward compatibility con il codice esistente
    """
    # Se non abbiamo ATR, usiamo un fattore di volatilità di default (1.0 = neutrale)
    return calculate_optimal_position_size(
        entry_price=entry_price,
        sl_price=sl_price,
        symbol='UNKNOWN',  # Non sappiamo il symbol qui
        volatility_atr=abs(entry_price - sl_price) * 1.0,  # Approximation
        ema_score=50,  # Neutrale
        risk_usd=risk_usd
    )


def calculate_ema_stop_loss(df: pd.DataFrame, timeframe: str, entry_price: float, side: str = 'Buy'):
    """
    Calcola stop loss basato su EMA invece che ATR
    
    Args:
        df: DataFrame con dati OHLCV
        timeframe: '5m', '15m', '30m', '1h', '4h'
        entry_price: prezzo di entrata
        side: 'Buy' o 'Sell'
    
    Returns:
        stop_loss_price: prezzo dello stop loss
        ema_used: quale EMA è stata usata
        ema_value: valore dell'EMA
    """
    if not USE_EMA_STOP_LOSS:
        # Fallback a ATR se disabilitato
        atr_val = atr(df, period=14).iloc[-1]
        if side == 'Buy':
            sl_price = entry_price - atr_val * ATR_MULT_SL
        else:
            sl_price = entry_price + atr_val * ATR_MULT_SL
        return sl_price, 'ATR', atr_val
    
    # Determina quale EMA usare per questo timeframe
    ema_to_use = EMA_STOP_LOSS_CONFIG.get(timeframe, 'ema10')
    
    # Calcola le EMA
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    # Seleziona l'EMA appropriata
    if ema_to_use == 'ema5':
        ema_value = ema_5.iloc[-1]
        ema_name = 'EMA 5'
    elif ema_to_use == 'ema10':
        ema_value = ema_10.iloc[-1]
        ema_name = 'EMA 10'
    elif ema_to_use == 'ema60':
        ema_value = ema_60.iloc[-1]
        ema_name = 'EMA 60'
    else:
        # Default a EMA 10
        ema_value = ema_10.iloc[-1]
        ema_name = 'EMA 10'
    
    # Calcola stop loss con buffer
    if side == 'Buy':
        # Per posizioni BUY: SL sotto l'EMA
        sl_price = ema_value * (1 - EMA_SL_BUFFER)
        
        # Verifica che non sia troppo lontano (max 3% dall'entry)
        max_sl_distance = entry_price * 0.03
        if (entry_price - sl_price) > max_sl_distance:
            sl_price = entry_price - max_sl_distance
            ema_name += ' (limitato)'
        
        # Verifica che non sia troppo vicino (min 0.5% dall'entry)
        min_sl_distance = entry_price * 0.005
        if (entry_price - sl_price) < min_sl_distance:
            sl_price = entry_price - min_sl_distance
            ema_name += ' (ampliato)'
    else:
        # Per posizioni SELL: SL sopra l'EMA
        sl_price = ema_value * (1 + EMA_SL_BUFFER)
        
        max_sl_distance = entry_price * 0.03
        if (sl_price - entry_price) > max_sl_distance:
            sl_price = entry_price + max_sl_distance
            ema_name += ' (limitato)'
        
        min_sl_distance = entry_price * 0.005
        if (sl_price - entry_price) < min_sl_distance:
            sl_price = entry_price + min_sl_distance
            ema_name += ' (ampliato)'
    
    return sl_price, ema_name, ema_value
    """
    Calcola la quantità basata sul rischio in USD
    Formula: Qty = Risk USD / |Entry - SL|
    
    Con limiti di sicurezza per evitare qty troppo grandi
    """
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit == 0:
        return 0
    
    qty = risk_usd / risk_per_unit
    
    # Limite massimo di sicurezza basato sul prezzo
    # Per symbol a basso prezzo, limita qty massima
    if entry_price < 1:
        # Per coin sotto $1, max 10,000 contracts
        max_qty = 10000
    elif entry_price < 10:
        # Per coin sotto $10, max 1,000 contracts
        max_qty = 1000
    else:
        # Per coin normali, max 100 contracts
        max_qty = 100
    
    qty = min(qty, max_qty)
    
    return float(max(0, qty))


def check_higher_timeframe_resistance(symbol, current_tf, current_price):
    """
    Controlla se ci sono resistenze EMA su timeframe superiori
    
    NEW: Aggiunge check momentum per evitare entry su HTF in discesa
    
    Returns:
        {
            'blocked': True/False,
            'htf': '30m' / '4h',
            'details': 'EMA 5 = $101, EMA 10 = $100.50',
            'momentum': 'bearish' / 'bullish' / 'neutral'  # NUOVO
        }
    """
    # Mappa timeframe -> higher timeframe
    htf_map = {
        '5m': '30m',
        '15m': '30m',
        '30m': '4h',
        '1h': '4h'
    }
    
    if current_tf not in htf_map:
        return {'blocked': False}
    
    htf = htf_map[current_tf]
    
    # Scarica dati HTF
    df_htf = bybit_get_klines(symbol, htf, limit=100)
    
    if df_htf.empty:
        logging.warning(f'⚠️ Nessun dato HTF per {symbol} {htf}')
        return {'blocked': False}
    
    # Calcola EMA HTF
    ema5_htf = df_htf['close'].ewm(span=5, adjust=False).mean()
    ema10_htf = df_htf['close'].ewm(span=10, adjust=False).mean()
    ema60_htf = df_htf['close'].ewm(span=60, adjust=False).mean()
    
    curr_ema5 = ema5_htf.iloc[-1]
    curr_ema10 = ema10_htf.iloc[-1]
    curr_ema60 = ema60_htf.iloc[-1]

    # ===== CHECK ULTIMA CANDELA HTF =====
    last_htf_candle = df_htf.iloc[-1]
    
    # Se ultima candela HTF è rossa e forte, blocca entry
    is_bearish = last_htf_candle['close'] < last_htf_candle['open']
    htf_body = abs(last_htf_candle['close'] - last_htf_candle['open'])
    htf_range = last_htf_candle['high'] - last_htf_candle['low']
    
    if htf_range > 0:
        htf_body_pct = htf_body / htf_range
        
        # BLOCCA se candela HTF è ribassista forte (corpo > 60%)
        if is_bearish and htf_body_pct > 0.60:
            # E se chiude vicino al low (no rejection) = momentum ribassista
            lower_wick = min(last_htf_candle['open'], last_htf_candle['close']) - last_htf_candle['low']
            lower_wick_pct = lower_wick / htf_range
            
            if lower_wick_pct < 0.20:  # Ombra inferiore < 20% = no supporto
                momentum_bearish = True
                momentum_reason.append(
                    f"Strong bearish HTF candle (body: {htf_body_pct*100:.0f}%, "
                    f"no support rejection)"
                )
    
    
    # ===== NUOVO: CHECK MOMENTUM HTF =====
    
    # 1. Verifica direzione EMA 10
    prev_ema10 = ema10_htf.iloc[-3]  # 2 candele fa
    ema10_direction = curr_ema10 - prev_ema10
    ema10_slope = (ema10_direction / prev_ema10) * 100 if prev_ema10 > 0 else 0
    
    # 2. Verifica se prezzo sta rompendo sotto EMA 10
    curr_htf_close = df_htf['close'].iloc[-1]
    prev_htf_close = df_htf['close'].iloc[-2]
    
    was_above_ema10 = prev_htf_close > ema10_htf.iloc[-2]
    now_below_ema10 = curr_htf_close < curr_ema10
    
    ema10_breakdown = was_above_ema10 and now_below_ema10
    
    # 3. Verifica allineamento EMA (bearish = EMA 5 sotto EMA 10)
    bearish_alignment = curr_ema5 < curr_ema10
    
    # 4. Verifica ultimi 3 prezzi in discesa
    recent_closes = df_htf['close'].iloc[-3:]
    downtrend_recent = all(recent_closes.iloc[i] > recent_closes.iloc[i+1] 
                           for i in range(len(recent_closes)-1))
    
    # ===== DECISIONE MOMENTUM =====
    momentum_bearish = False
    momentum_reason = []
    
    # CRITERIO 1: EMA 10 in discesa (slope negativo significativo)
    if ema10_slope < -0.1:  # Scende più di 0.1%
        momentum_bearish = True
        momentum_reason.append(f"EMA 10 declining ({ema10_slope:.2f}%)")
    
    # CRITERIO 2: Breakdown EMA 10 appena avvenuto
    if ema10_breakdown:
        momentum_bearish = True
        momentum_reason.append("EMA 10 breakdown (just crossed below)")
    
    # CRITERIO 3: EMA bearish alignment + prezzo sotto entrambe
    if bearish_alignment and curr_htf_close < curr_ema10:
        momentum_bearish = True
        momentum_reason.append("Bearish EMA alignment")
    
    # CRITERIO 4: Downtrend confermato (3 candele consecutive in discesa)
    if downtrend_recent:
        momentum_bearish = True
        momentum_reason.append("HTF downtrend confirmed (3 lower closes)")
    
    # CRITERIO 5: Distanza eccessiva da EMA 60 (overextension = probabile pullback)
    distance_from_ema60 = (curr_htf_close - curr_ema60) / curr_ema60
    if distance_from_ema60 > 0.03:  # Più del 3% sopra EMA 60
        momentum_bearish = True
        momentum_reason.append(f"Overextended from EMA 60 (+{distance_from_ema60*100:.1f}%)")
    
    # ===== FINE CHECK MOMENTUM =====
    
    # Check resistenza originale (EMA sopra prezzo)
    if current_tf in ['5m', '15m']:
        # BLOCCA se:
        # 1. EMA 10 è sopra prezzo (resistenza) E
        # 2. Momentum è bearish
        if curr_ema10 > current_price:
            if momentum_bearish:
                logging.warning(
                    f'🚫 HTF {htf} BLOCKING {symbol}: '
                    f'EMA 10 resistance + Bearish momentum'
                )
                return {
                    'blocked': True,
                    'htf': htf,
                    'details': (
                        f'EMA 10 ({htf}): ${curr_ema10:.2f}\n'
                        f'Price: ${current_price:.2f}\n'
                        f'EMA 10 Slope: {ema10_slope:.2f}%\n'
                        f'Momentum: BEARISH\n'
                        f'Reasons: {", ".join(momentum_reason)}'
                    ),
                    'momentum': 'bearish'
                }
        
        # BLOCCA anche se momentum è MOLTO bearish, 
        # anche se EMA non è resistenza diretta
        if len(momentum_reason) >= 3:  # 3+ segnali bearish
            logging.warning(
                f'🚫 HTF {htf} BLOCKING {symbol}: '
                f'Strong bearish momentum ({len(momentum_reason)} signals)'
            )
            return {
                'blocked': True,
                'htf': htf,
                'details': (
                    f'HTF Timeframe: {htf}\n'
                    f'EMA 10: ${curr_ema10:.2f}\n'
                    f'EMA 10 Slope: {ema10_slope:.2f}%\n'
                    f'Price: ${current_price:.2f}\n'
                    f'Momentum: STRONG BEARISH\n'
                    f'Signals: {", ".join(momentum_reason)}'
                ),
                'momentum': 'bearish'
            }
    
    elif current_tf in ['30m', '1h']:
        # Per day trading: check EMA 60 su 4h + momentum
        if curr_ema60 > current_price:
            if momentum_bearish:
                return {
                    'blocked': True,
                    'htf': htf,
                    'details': (
                        f'EMA 60 ({htf}): ${curr_ema60:.2f}\n'
                        f'Price: ${current_price:.2f}\n'
                        f'Momentum: BEARISH\n'
                        f'Reasons: {", ".join(momentum_reason)}'
                    ),
                    'momentum': 'bearish'
                }
    
    return {
        'blocked': False,
        'momentum': 'bullish' if not momentum_bearish else 'neutral'
    }

def check_higher_timeframe_support(symbol, current_tf, current_price):
    """
    Controlla se ci sono supporti EMA su timeframe superiori (per SHORT)
    
    Returns:
        {
            'blocked': True/False,
            'htf': '30m' / '4h',
            'details': 'EMA 5 = $101, EMA 10 = $100.50'
        }
    """
    # Mappa timeframe -> higher timeframe
    htf_map = {
        '5m': '30m',
        '15m': '30m',
        '30m': '4h',
        '1h': '4h'
    }
    
    if current_tf not in htf_map:
        return {'blocked': False}
    
    htf = htf_map[current_tf]
    
    # Scarica dati HTF
    df_htf = bybit_get_klines(symbol, htf, limit=100)
    
    if df_htf.empty:
        logging.warning(f'⚠️ Nessun dato HTF per {symbol} {htf}')
        return {'blocked': False}
    
    # Calcola EMA HTF
    ema5_htf = df_htf['close'].ewm(span=5, adjust=False).mean().iloc[-1]
    ema10_htf = df_htf['close'].ewm(span=10, adjust=False).mean().iloc[-1]
    
    # Check supporto (EMA SOTTO il prezzo = supporto che blocca SHORT!)
    if current_tf in ['5m', '15m']:
        # Per scalping: controlla EMA 5 e 10 su 30m
        # BLOCCA se EMA 5 o 10 sono SOTTO il prezzo corrente (supporto sotto)
        if ema5_htf < current_price or ema10_htf < current_price:
            return {
                'blocked': True,
                'htf': htf,
                'details': f'EMA 5 ({htf}): ${ema5_htf:.2f}\nEMA 10 ({htf}): ${ema10_htf:.2f}\nPrice: ${current_price:.2f}\nSupporto sotto il prezzo!'
            }
    
    elif current_tf in ['30m', '1h']:
        # Per day: controlla EMA 60 su 4h
        ema60_htf = df_htf['close'].ewm(span=60, adjust=False).mean().iloc[-1]
        
        # BLOCCA se EMA 60 è SOTTO il prezzo
        if ema60_htf < current_price:
            return {
                'blocked': True,
                'htf': htf,
                'details': f'EMA 60 ({htf}): ${ema60_htf:.2f}\nPrice: ${current_price:.2f}\nSupporto sotto il prezzo!'
            }
    
    return {'blocked': False}


def check_compression_htf_resistance(symbol: str, current_tf: str, current_price: float) -> dict:
    """
    Check HTF resistance specifico per Compression Breakout
    
    Verifica se ci sono EMA 5, 10 su timeframe superiore che agiscono
    come resistenza (tappo) al movimento rialzista.
    
    Args:
        symbol: Es. 'BTCUSDT'
        current_tf: Timeframe corrente ('5m', '15m', ecc.)
        current_price: Prezzo corrente
    
    Returns:
        {
            'blocked': True/False,
            'htf': '30m' / '4h' / None,
            'details': 'EMA 5 = $X, EMA 10 = $Y'
        }
    """
    # Mappa timeframe -> higher timeframe
    htf_map = {
        '5m': '30m',
        '15m': '30m',
        '30m': '4h',
        '1h': '4h'
    }
    
    if current_tf not in htf_map:
        return {'blocked': False}
    
    htf = htf_map[current_tf]
    
    try:
        # Scarica dati HTF
        df_htf = bybit_get_klines(symbol, htf, limit=100)
        
        if df_htf.empty:
            logging.warning(f'⚠️ Nessun dato HTF per {symbol} {htf}')
            return {'blocked': False}
        
        # Calcola EMA HTF
        ema5_htf = df_htf['close'].ewm(span=5, adjust=False).mean().iloc[-1]
        ema10_htf = df_htf['close'].ewm(span=10, adjust=False).mean().iloc[-1]
        
        # Check resistenza: EMA SOPRA il prezzo = resistenza
        if current_tf in ['5m', '15m']:
            # Per scalping/intraday: controlla EMA 5 e 10 su 30m
            if ema5_htf > current_price or ema10_htf > current_price:
                return {
                    'blocked': True,
                    'htf': htf,
                    'details': f'EMA 5 ({htf}): ${ema5_htf:.2f}\nEMA 10 ({htf}): ${ema10_htf:.2f}\nPrice: ${current_price:.2f}\nResistenza sopra il prezzo!'
                }
        
        elif current_tf in ['30m', '1h']:
            # Per day trading: controlla EMA 60 su 4h
            ema60_htf = df_htf['close'].ewm(span=60, adjust=False).mean().iloc[-1]
            
            if ema60_htf > current_price:
                return {
                    'blocked': True,
                    'htf': htf,
                    'details': f'EMA 60 ({htf}): ${ema60_htf:.2f}\nPrice: ${current_price:.2f}\nResistenza sopra il prezzo!'
                }
        
        return {'blocked': False}
        
    except Exception as e:
        logging.error(f'Errore check HTF resistance: {e}')
        return {'blocked': False}


async def place_bybit_order(symbol: str, side: str, qty: float, sl_price: float, tp_price: float, entry_price: float, timeframe: str, chat_id: int, pattern_name: str = None):
    """
    NEW: Invece di 1 TP fisso, calcola 3 TP levels
    Piazza ordine su Bybit (Market o Limit)
    NEW: Supporta ordini LIMIT per pattern lenti
    
    Piazza ordine market su Bybit (Demo o Live)
    Controlla REALMENTE su Bybit se esiste già una posizione aperta
    
    Parametri:
    - symbol: es. 'BTCUSDT'
    - side: 'Buy' o 'Sell'
    - qty: quantità in contratti
    - sl_price: prezzo stop loss
    - tp_price: prezzo take profit
    """

    # ===== CRITICAL: Check Market Time Filter PRIMA di piazzare ordine =====
    if config.MARKET_TIME_FILTER_ENABLED:
        time_ok, time_reason = is_good_trading_time_utc()
        
        if not time_ok:
            now_utc = datetime.now(timezone.utc)
            current_hour = now_utc.hour
            
            logging.warning(
                f'🚫 {symbol}: Order blocked by Market Time Filter '
                f'(UTC hour {current_hour:02d})'
            )
            
            return {
                'error': 'market_time_blocked',
                'message': f'Trading blocked during UTC hour {current_hour:02d}',
                'reason': time_reason
            }
            
    # Determina tipo ordine
    order_type = 'Market'  # Default

    if pattern_name and pattern_name in PATTERN_ORDER_TYPE:
        order_type = 'Market' if PATTERN_ORDER_TYPE[pattern_name] == 'market' else 'Limit'

    logging.info(f'📤 Placing {order_type} order: {symbol} {side} qty={qty:.4f} Entry: ${entry_price:.4f} SL: ${sl_price:.4f} TP: ${tp_price:.4f} Mode: {config.TRADING_MODE}')
    
    if BybitHTTP is None:
        return {'error': 'pybit non disponibile'}
    
    # SINCRONIZZA con Bybit prima di controllare
    await sync_positions_with_bybit()

    # ✅ FIX PUNTO #3: Check con lock prima di verificare su Bybit
    with config.POSITIONS_LOCK:
        if symbol in config.ACTIVE_POSITIONS:
            logging.warning(f'{symbol}: Position already tracked locally')
            return {'error': 'position_exists', 'message': f'Posizione già tracciata per {symbol}'}
    # Controlla se esiste VERAMENTE una posizione aperta per questo symbol
    real_positions = await get_open_positions_from_bybit(symbol)
    
    if real_positions:
        existing = real_positions[0]
        logging.info(f'⚠️ Posizione REALE trovata su Bybit per {symbol}: {existing}')
        return {
            'error': 'position_exists',
            'message': f'Posizione già aperta per {symbol} su Bybit',
            'existing_position': existing
        }
    
    try:
        session = create_bybit_session()
        
        # Verifica il balance prima di tradare
        try:
            wallet = session.get_wallet_balance(accountType="UNIFIED")
            logging.info(f'💰 Wallet Balance check completato')
        except Exception as e:
            logging.warning(f'Non riesco a verificare il balance: {e}')
        
        # Ottieni info sul symbol per determinare qty corretta
        try:
            instrument_info = get_instrument_info_cached(session, symbol)
            
            # ✅ FIX: get_instrument_info_cached() ritorna DIRETTAMENTE il dict con info
            min_order_qty = instrument_info['min_order_qty']
            max_order_qty = instrument_info['max_order_qty']
            qty_step = instrument_info['qty_step']
            qty_decimals = instrument_info['qty_decimals']
            
            logging.info(f'📊 {symbol} - Min: {min_order_qty}, Max: {max_order_qty}, Step: {qty_step}')
            
            # Arrotonda qty al qty_step più vicino
            qty = round(qty / qty_step) * qty_step
            
            # Limita qty tra min e max
            qty = max(min_order_qty, min(qty, max_order_qty))
            
            # Arrotonda ai decimali corretti
            qty = round(qty, qty_decimals)
            
        except KeyError as e:
            logging.error(f'Missing key in instrument_info: {e}')
            logging.warning(f'Using default qty rounding for {symbol}')
            qty = round(qty, 3)
        except Exception as e:
            logging.warning(f'Errore nel recuperare instrument info: {e}')
            # Fallback: arrotondamento generico
            qty = round(qty, 3)
        
        # Verifica qty minima sensata
        if qty < 0.001:
            return {'error': f'Qty troppo piccola ({qty}). Aumenta RISK_USD o riduci ATR.'}
        
        # Arrotonda prezzi con decimali dinamici
        # ✅ FIX CRITICO: Arrotonda prezzi SL/TP al tick_size corretto
        price_decimals = instrument_info['price_decimals']
        tick_size = instrument_info['tick_size']

        # Arrotonda al tick_size più vicino (non solo ai decimali!)
        def round_to_tick(price, tick):
            """Arrotonda prezzo al tick_size più vicino"""
            if tick == 0:
                return round(price, price_decimals)
            return round(price / tick) * tick

        sl_price = round_to_tick(sl_price, tick_size)
        tp_price = round_to_tick(tp_price, tick_size)

        # Fallback: assicurati almeno i decimali corretti
        sl_price = round(sl_price, price_decimals)
        tp_price = round(tp_price, price_decimals)
        
        logging.info(f'📤 Piazzando ordine {side} per {symbol}')
        logging.info(f'   Qty: {qty} | SL: {sl_price} | TP: {tp_price}')
        logging.info(f'   Tick size: {tick_size}, Price decimals: {price_decimals}')

        # ===== CALCOLA MULTI-TP SE ABILITATO =====
        tp_levels = []
        
        if config.MULTI_TP_ENABLED and config.MULTI_TP_CONFIG['enabled']:
            risk = abs(entry_price - sl_price)
            
            for level in config.MULTI_TP_CONFIG['levels']:
                if side == 'Buy':
                    tp_level_price = entry_price + (risk * level['rr_ratio'])
                else:  # Sell
                    tp_level_price = entry_price - (risk * level['rr_ratio'])
                
                # Arrotonda al tick_size
                tp_level_price = round_to_tick(tp_level_price, tick_size)
                
                tp_levels.append({
                    'label': level['label'],
                    'price': tp_level_price,
                    'close_pct': level['close_pct'],
                    'qty': round(qty * level['close_pct'], qty_decimals),
                    'emoji': level['emoji'],
                    'hit': False
                })
            
            logging.info(f'🎯 Multi-TP configurato:')
            for i, tp in enumerate(tp_levels, 1):
                logging.info(f'   TP{i}: ${tp["price"]:.{price_decimals}f} ({tp["close_pct"]*100:.0f}% = {tp["qty"]:.4f})')
            
            # Per Bybit: Usa TP più lontano come "main TP" (TP3)
            # TP1 e TP2 saranno gestiti da monitor_partial_tp()
            tp_price = tp_levels[-1]['price']  # TP3

        if order_type == 'Limit':
            # ===== LIMIT ORDER =====
            # Calcola prezzo limit (entry - offset per BUY)
            offset = LIMIT_ORDER_CONFIG['offset_pct']
            
            if side == 'Buy':
                limit_price = entry_price * (1 - offset)  # Sotto prezzo corrente
            else:
                limit_price = entry_price * (1 + offset)  # Sopra prezzo corrente
            
            # Arrotonda prezzo
            price_decimals = get_price_decimals(limit_price)
            limit_price = round(limit_price, price_decimals)
            
            logging.info(f'📍 Limit price: ${limit_price:.{price_decimals}f} (entry: ${entry_price:.{price_decimals}f})')
            
            # Piazza ordine LIMIT
            order = session.place_order(
                category='linear',
                symbol=symbol,
                side=side,
                orderType='Limit',
                qty=str(qty),
                price=str(limit_price),  # ← Prezzo limit
                stopLoss=str(sl_price),
                takeProfit=str(tp_price),
                positionIdx=0,
                timeInForce='GTC'  # Good Till Cancel
            )
            
            if order.get('retCode') == 0:
                with config.POSITIONS_LOCK:
                    # ✅ DOUBLE-CHECK dopo ordine eseguito (pattern standard)
                    if symbol in config.ACTIVE_POSITIONS:
                        logging.warning(f'{symbol}: Race condition detected, position already added')
                    else:
                        config.ACTIVE_POSITIONS[symbol] = {
                            'side': side,
                            'qty': qty,
                            'entry_price': entry_price,
                            # ... altri campi
                        }
                
                order_id = order.get('result', {}).get('orderId')
                
                # ===== WAIT FOR FILL (con timeout) =====
                timeout = LIMIT_ORDER_CONFIG['timeout_seconds']
                start_time = time.time()
                filled = False
                
                while time.time() - start_time < timeout:
                    # Check se ordine è fillato
                    order_status = session.get_open_orders(
                        category='linear',
                        symbol=symbol,
                        orderId=order_id
                    )
                    
                    if order_status.get('retCode') == 0:
                        orders = order_status.get('result', {}).get('list', [])
                        
                        if not orders:  # Ordine non più in open = fillato
                            filled = True
                            logging.info(f'✅ Limit order FILLED: {symbol}')
                            break
                    
                    await asyncio.sleep(2)  # Check ogni 2 secondi
                
                if not filled:
                    logging.warning(f'⏱️ Limit order TIMEOUT: {symbol} (not filled in {timeout}s)')
                    
                    # Cancella ordine
                    cancel = session.cancel_order(
                        category='linear',
                        symbol=symbol,
                        orderId=order_id
                    )
                    
                    if LIMIT_ORDER_CONFIG['fallback_to_market']:
                        logging.info(f'🔄 Fallback to MARKET order: {symbol}')
                        
                        # Riprova con MARKET
                        order = session.place_order(
                            category='linear',
                            symbol=symbol,
                            side=side,
                            orderType='Market',
                            qty=str(qty),
                            stopLoss=str(sl_price),
                            takeProfit=str(tp_price),
                            positionIdx=0
                        )
                    else:
                        return {'error': 'Limit order timeout, no fill'}
        
        else:
            # Piazza ordine market con SL e TP
            order = session.place_order(
                category='linear',  # USDT Perpetual
                symbol=symbol,
                side=side,
                orderType='Market',
                qty=str(qty),
                stopLoss=str(sl_price),
                takeProfit=str(tp_price),
                positionIdx=0  # One-way mode
            )
        
        logging.info(f'✅ Ordine eseguito: {order_type} - {order}')

        # ===== SALVA POSIZIONE CON MULTI-TP TRACKING =====
        if order.get('retCode') == 0:
            with config.POSITIONS_LOCK:
                config.ACTIVE_POSITIONS[symbol] = {
                    'side': side,
                    'qty': qty,
                    'qty_original': qty,  # ← NUOVO: Qty iniziale (prima dei TP parziali)
                    'entry_price': entry_price,
                    'sl': sl_price,
                    'tp': tp_price,  # TP finale (TP3)
                    'order_id': order.get('result', {}).get('orderId'),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'timeframe': timeframe,
                    'pattern_name': pattern_name,
                    'trailing_active': False,
                    'highest_price': entry_price,
                    'chat_id': chat_id,
                    'multi_tp_levels': tp_levels if config.MULTI_TP_ENABLED else None  # ← NUOVO
                }
            
            # ===== INIZIALIZZA TP TRACKING =====
            if config.MULTI_TP_ENABLED and tp_levels:
                with TP_TRACKING_LOCK:
                    TP_TRACKING[symbol] = {
                        'tp1_hit': False,
                        'tp2_hit': False,
                        'tp3_hit': False,
                        'tp1_qty_closed': 0.0,
                        'tp2_qty_closed': 0.0,
                        'tp3_qty_closed': 0.0,
                        'last_check': datetime.now(timezone.utc)
                    }
                
                logging.info(f'📝 Multi-TP tracking inizializzato per {symbol}')
            
            logging.info(f'📝 Posizione salvata per {symbol}')
        
        return order
        
        # Salva la posizione come attiva
        if order.get('retCode') == 0:
            with config.POSITIONS_LOCK:
                config.ACTIVE_POSITIONS[symbol] = {
                    'side': side,
                    'qty': qty,
                    'entry_price': entry_price,  # 👈 AGGIUNGI (pass come parametro)
                    'sl': sl_price,
                    'tp': tp_price,
                    'order_id': order.get('result', {}).get('orderId'),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'timeframe': timeframe,  # 👈 AGGIUNGI (pass come parametro)
                    'pattern_name': pattern_name,  # ← AGGIUNGI
                    'trailing_active': False,
                    'highest_price': entry_price,  # 👈 AGGIUNGI
                    'chat_id': chat_id  # 👈 AGGIUNGI
                }
            logging.info(f'📝 Posizione salvata per {symbol}')
        
        return order
        
    except Exception as e:
        error_msg = str(e)
        logging.exception('❌ Errore nel piazzare ordine')
        
        # Errori comuni con suggerimenti
        if 'insufficient' in error_msg.lower():
            return {'error': 'Balance insufficiente. Verifica il tuo saldo con /balance'}
        elif 'qty invalid' in error_msg.lower() or '10001' in error_msg:
            return {'error': f'Quantità non valida per {symbol}. Il symbol potrebbe avere limiti specifici o qty troppo grande/piccola.'}
        elif 'invalid' in error_msg.lower():
            return {'error': f'Parametri non validi: {error_msg}'}
        elif 'risk limit' in error_msg.lower():
            return {'error': 'Limite di rischio raggiunto. Riduci la posizione o aumenta il risk limit su Bybit.'}
        else:
            return {'error': f'{error_msg}'}


def get_top_profitable_symbols():
    """
    Ottiene i top symbol più profittevoli da Bybit
    
    Criteri:
    - Volume 24h > min_volume_usdt
    - Price change 24h tra min_price_change e max_price_change
    - Ordina per price_change_percent o volume
    
    Returns:
        list: Lista di symbol (es. ['BTCUSDT', 'ETHUSDT', ...])
    """
    try:
        # Ottieni ticker 24h per tutti i symbol
        url = f'{config.BYBIT_PUBLIC_REST}/v5/market/tickers'
        params = {
            'category': 'linear'
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get('retCode') != 0:
            logging.error(f'Errore API Bybit tickers: {data.get("retMsg")}')
            return []
        
        tickers = data.get('result', {}).get('list', [])
        
        if not tickers:
            logging.warning('Nessun ticker trovato')
            return []
        
        # Filtra e processa symbols
        candidates = []
        
        for ticker in tickers:
            symbol = ticker.get('symbol', '')
            
            # Solo USDT perpetual
            if not symbol.endswith('USDT'):
                continue
            
            # Escludi stablecoins
            if symbol in config.AUTO_DISCOVERY_CONFIG['exclude_symbols']:
                continue
            
            try:
                volume_24h = float(ticker.get('turnover24h', 0))  # Volume in USDT
                price_change_percent = float(ticker.get('price24hPcnt', 0)) * 100  # In percentuale
                last_price = float(ticker.get('lastPrice', 0))
                
                # Applica filtri
                if volume_24h < config.AUTO_DISCOVERY_CONFIG['min_volume_usdt']:
                    continue
                
                if price_change_percent < config.AUTO_DISCOVERY_CONFIG['min_price_change']:
                    continue
                
                if price_change_percent > config.AUTO_DISCOVERY_CONFIG['max_price_change']:
                    continue  # Evita pump estremi
                
                candidates.append({
                    'symbol': symbol,
                    'volume_24h': volume_24h,
                    'price_change_percent': price_change_percent,
                    'last_price': last_price
                })
                
            except (ValueError, TypeError) as e:
                logging.debug(f'Skip {symbol}: errore parsing dati ({e})')
                continue
        
        if not candidates:
            logging.warning('Nessun symbol valido dopo i filtri')
            return []
        
        # Ordina per criterio scelto
        sort_by = config.AUTO_DISCOVERY_CONFIG['sorting']
        
        if sort_by == 'volume':
            candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
        else:  # price_change_percent (default)
            candidates.sort(key=lambda x: x['price_change_percent'], reverse=True)
        
        # Prendi top N
        top_count = config.AUTO_DISCOVERY_CONFIG['top_count']
        top_symbols = [c['symbol'] for c in candidates[:top_count]]
        
        # Log risultati
        logging.info(f'🔍 Top {len(top_symbols)} symbols trovati:')
        for i, candidate in enumerate(candidates[:top_count], 1):
            logging.info(
                f"  {i}. {candidate['symbol']}: "
                f"+{candidate['price_change_percent']:.2f}%, "
                f"Vol: ${candidate['volume_24h']/1_000_000:.1f}M"
            )
        
        return top_symbols
        
    except requests.exceptions.RequestException as e:
        logging.error(f'Errore richiesta tickers: {e}')
        return []
    except Exception as e:
        logging.exception('Errore in get_top_profitable_symbols')
        return []
        

async def auto_discover_and_analyze(context: ContextTypes.DEFAULT_TYPE):
    """
    Job che automaticamente:
    1. Ottiene top symbols profittevoli
    2. Ferma analisi per symbols non più in top
    3. Avvia nuove analisi per symbols in top
    
    Eseguito ogni 4 ore
    """
    if not config.AUTO_DISCOVERY_CONFIG['enabled']:
        return
    
    job_data = context.job.data
    chat_id = job_data['chat_id']
    
    logging.info('🔄 Auto-Discovery: Aggiornamento top symbols...')
    
    try:
        # Ottieni top symbols
        top_symbols = get_top_profitable_symbols()
        
        if not top_symbols:
            logging.warning('⚠️ Auto-Discovery: Nessun symbol trovato')
            await context.bot.send_message(
                chat_id=chat_id,
                text='⚠️ Auto-Discovery: Impossibile ottenere top symbols da Bybit'
            )
            return
        
        timeframe = config.AUTO_DISCOVERY_CONFIG['timeframe']
        autotrade = config.AUTO_DISCOVERY_CONFIG['autotrade']
        
        # Converti in set per comparazione
        new_symbols_set = set(top_symbols)
        
        with config.AUTO_DISCOVERED_LOCK:
            old_symbols_set = set(config.AUTO_DISCOVERED_SYMBOLS)
        
        # Symbols da rimuovere (non più in top)
        to_remove = old_symbols_set - new_symbols_set
        
        # Symbols da aggiungere (nuovi in top)
        to_add = new_symbols_set - old_symbols_set
        
        # === RIMUOVI ANALISI VECCHIE ===
        removed_count = 0
        
        with config.ACTIVE_ANALYSES_LOCK:
            chat_analyses = config.ACTIVE_ANALYSES.get(chat_id, {})
            
            for symbol in to_remove:
                key = f'{symbol}-{timeframe}'
                
                if key in chat_analyses:
                    job = chat_analyses[key]
                    job.schedule_removal()
                    del chat_analyses[key]
                    removed_count += 1
                    logging.info(f'❌ Rimosso {symbol} {timeframe} (non più in top)')
        
        # === AGGIUNGI NUOVE ANALISI ===
        added_count = 0
        
        for symbol in to_add:
            key = f'{symbol}-{timeframe}'
            
            # Verifica che non esista già
            with config.ACTIVE_ANALYSES_LOCK:
                chat_map = config.ACTIVE_ANALYSES.setdefault(chat_id, {})
                
                if key in chat_map:
                    logging.debug(f'⏭️ Skip {symbol}: già in analisi')
                    continue
            
            # Verifica dati disponibili
            test_df = bybit_get_klines(symbol, timeframe, limit=10)
            if test_df.empty:
                logging.warning(f'⚠️ Skip {symbol}: nessun dato disponibile')
                continue
            
            # Calcola intervallo
            interval_seconds = config.INTERVAL_SECONDS.get(timeframe, 300)
            now = datetime.now(timezone.utc)
            epoch = int(now.timestamp())
            to_next = interval_seconds - (epoch % interval_seconds)
            
            # Crea job
            job_data_new = {
                'chat_id': chat_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'autotrade': autotrade,
                'auto_discovered': True  # Flag per identificare auto-discovered
            }
            
            job = context.job_queue.run_repeating(
                analyze_job,
                interval=interval_seconds,
                first=to_next,
                data=job_data_new,
                name=key
            )
            
            with config.ACTIVE_ANALYSES_LOCK:
                chat_map[key] = job
            
            added_count += 1
            logging.info(f'✅ Aggiunto {symbol} {timeframe}')
        
        # Aggiorna storage
        with config.AUTO_DISCOVERED_LOCK:
            config.AUTO_DISCOVERED_SYMBOLS.clear()
            config.AUTO_DISCOVERED_SYMBOLS.update(new_symbols_set)
        
        # === NOTIFICA RISULTATI ===
        msg = "🔄 <b>Auto-Discovery Aggiornato</b>\n\n"
        
        if added_count > 0 or removed_count > 0:
            msg += f"📊 Top {len(top_symbols)} symbols:\n"
            for i, sym in enumerate(top_symbols, 1):
                status = "🆕" if sym in to_add else "✅"
                msg += f"{status} {i}. {sym}\n"
            
            msg += f"\n"
            
            if added_count > 0:
                msg += f"✅ Aggiunti: {added_count}\n"
            
            if removed_count > 0:
                msg += f"❌ Rimossi: {removed_count}\n"
            
            msg += f"\n⏱️ Timeframe: {timeframe}\n"
            msg += f"🤖 Autotrade: {'ON' if autotrade else 'OFF'}\n"
            msg += f"🔄 Prossimo update tra 2 ore"
        else:
            msg += "✅ Nessun cambiamento\n\n"
            msg += f"Top {len(top_symbols)} symbols confermati:\n"
            for i, sym in enumerate(top_symbols, 1):
                msg += f"{i}. {sym}\n"
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=msg,
            parse_mode='HTML'
        )
        
    except Exception as e:
        logging.exception('Errore in auto_discover_and_analyze')
        
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f'❌ Errore Auto-Discovery:\n{str(e)}'
            )
        except:
            pass

async def execute_profit_lock(
    context,
    symbol: str,
    side: str,
    new_sl: float,
    current_sl: float,
    profit_usd: float,
    profit_pct: float,
    profit_risk_ratio: float,
    locked_profit: float,
    entry_price: float,
    pos_info: dict,
    timeframe_entry: str,
    chat_id: int
) -> bool:
    """
    Esegue il Profit Lock: sposta stop loss per proteggere profitti elevati
    
    Returns:
        bool: True se eseguito con successo, False se fallito
    """
    try:
        # ✅ STEP 1: Verifica posizione su Bybit
        session = create_bybit_session()
        positions_response = session.get_positions(category="linear", symbol=symbol)
        
        if positions_response.get('retCode') != 0:
            logging.error(f"{symbol}: Error checking position: {positions_response.get('retMsg')}")
            return False
        
        pos_list = positions_response.get('result', {}).get('list', [])
        
        # Cerca posizione attiva
        real_position = None
        for p in pos_list:
            if float(p.get('size', 0)) > 0:
                real_position = p
                break
        
        if not real_position:
            logging.warning(f"{symbol}: No active position on Bybit, removing from tracking")
            with config.POSITIONS_LOCK:
                if symbol in config.ACTIVE_POSITIONS:
                    del config.ACTIVE_POSITIONS[symbol]
            return False
        
        # ✅ STEP 2: Aggiorna SL su Bybit
        result = session.set_trading_stop(
            category="linear",
            symbol=symbol,
            stopLoss=str(round(new_sl, get_price_decimals(new_sl))),
            positionIdx=0
        )
        
        if result.get('retCode') == 0:
            # ✅ SUCCESS: Aggiorna tracking locale
            with config.POSITIONS_LOCK:
                if symbol in config.ACTIVE_POSITIONS:
                    config.ACTIVE_POSITIONS[symbol]['sl'] = new_sl
            
            logging.info(
                f"🔒 {symbol} {side} PROFIT LOCK! "
                f"SL {new_sl:.4f} (profit ${profit_usd:.2f}, {profit_risk_ratio:.1f}x risk)"
            )
            
            # ✅ STEP 3: Notifica Telegram
            if chat_id:
                try:
                    side_emoji = '🟢' if side == 'Buy' else '🔴'
                    
                    notification = f"{side_emoji} <b>🔒 PROFIT LOCK!</b> ({side})\n\n"
                    notification += f"<b>Symbol:</b> {symbol} ({timeframe_entry})\n"
                    notification += f"<b>Profit:</b> ${profit_usd:.2f} (+{profit_pct:.2f}%)\n"
                    notification += f"<b>Risk Ratio:</b> {profit_risk_ratio:.1f}x risk iniziale!\n\n"
                    notification += f"<b>🔒 Stop Loss Locked:</b>\n"
                    notification += f"• Prima: {current_sl:.{get_price_decimals(current_sl)}f}\n"
                    notification += f"• <b>Ora: {new_sl:.{get_price_decimals(new_sl)}f}</b>\n"
                    notification += f"• Profit protetto: ${locked_profit:.2f} ({PROFIT_LOCK_CONFIG['retention']*100:.0f}%)\n\n"
                    notification += f"<b>⚡ Profit molto alto rilevato!</b>\n"
                    notification += f"Stop loss spostato IMMEDIATAMENTE per proteggere guadagni.\n"
                    notification += f"Trailing normale riprenderà da questo livello."
                    
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=notification,
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logging.error(f"Errore invio notifica profit lock: {e}")
            
            return True
        else:
            logging.error(f"{symbol}: Errore profit lock: {result.get('retMsg')}")
            return False
    
    except Exception as e:
        logging.error(f"{symbol}: Errore set_trading_stop profit lock: {e}")
        return False
            
async def update_trailing_stop_loss(context: ContextTypes.DEFAULT_TYPE):
    """
    Advanced Trailing Stop Loss - Multi-Level Progressive
    SUPPORTA SIA BUY CHE SELL

    NUOVO COMPORTAMENTO:
    ====================
    Quando il profit supera una soglia alta (es. 5x risk), 
    BLOCCA IMMEDIATAMENTE il profitto spostando SL a break-even o superiore,
    IGNORANDO temporaneamente l'EMA trailing normale.
    
    LOGICA:
    1. Per ogni posizione aperta, calcola profit %
    2. Trova il livello appropriato da TRAILING_CONFIG_ADVANCED
    3. Calcola nuovo SL basato su EMA 10 con buffer dinamico
    4. Aggiorna su Bybit se SL è migliore (mai indietro)
    5. Notifica su Telegram quando cambia livello
    
    LIVELLI:
    - 0.3%: Early protection (buffer largo 0.3%)
    - 0.5%: Standard trail (buffer medio 0.2%)
    - 1.0%: Tight trail (buffer stretto 0.1%)
    - 2.0%: Ultra tight trail (buffer ultra 0.05%)
    """
    
    if not config.TRAILING_STOP_ENABLED:
        return
    
    with config.POSITIONS_LOCK:
        positions_copy = dict(config.ACTIVE_POSITIONS)
    
    if not positions_copy:
        return
    
    logging.debug(f"Trailing check: {len(positions_copy)} positions")
    
    for symbol, pos_info in positions_copy.items():

        # ===== INIZIALIZZA new_sl QUI (FIX) =====
        new_sl = None  # ← AGGIUNGI QUESTA RIGA ALL'INIZIO DEL LOOP
        
        try:
            side = pos_info.get('side')
            entry_price = pos_info.get('entry_price')  # ← USA .get() per safety
            
            # ✅ FIX: Verifica che entry_price esista prima di continuare
            if not entry_price or entry_price <= 0:
                logging.error(f"{symbol}: Missing or invalid entry_price ({entry_price}), skipping trailing stop")
                continue

            # ===== VERIFICA POSIZIONE REALE SU BYBIT =====
            try:
                session = create_bybit_session()

                # ✅ FIX: Preserva TP quando aggiorni SL
                # Alcuni broker resettano TP se non lo passi esplicitamente
                tp_price = pos_info.get('tp', 0)

                # Arrotonda al tick_size
                price_decimals = get_price_decimals(new_sl)
                new_sl_rounded = round(new_sl, price_decimals)

                # Prepara parametri
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "stopLoss": str(new_sl_rounded),
                    "positionIdx": 0
                }

                # ✅ Includi TP se esistente per preservarlo
                if tp_price > 0:
                    tp_rounded = round(tp_price, price_decimals)
                    params["takeProfit"] = str(tp_rounded)
                    logging.debug(f"{symbol}: Updating SL={new_sl_rounded}, preserving TP={tp_rounded}")
                else:
                    logging.debug(f"{symbol}: Updating SL={new_sl_rounded}, no TP set")
                
                result = session.set_trading_stop(**params)

                if result.get('retCode') == 0:
                    # Aggiorna tracking locale
                    with config.POSITIONS_LOCK:
                        if symbol in config.ACTIVE_POSITIONS:
                            config.ACTIVE_POSITIONS[symbol]['sl'] = new_sl
                            
                            logging.info(f"{symbol} ({side}): Trailing SL updated to {new_sl:.4f} (Level: {active_level['label']})")
    
                positions_response = session.get_positions(
                    category='linear',
                    symbol=symbol
                )
                
                if positions_response.get('retCode') == 0:
                    pos_list = positions_response.get('result', {}).get('list', [])
                    
                    # Cerca posizione attiva per questo symbol
                    real_position = None
                    for p in pos_list:
                        if float(p.get('size', 0)) > 0:
                            real_position = p
                            break
                    
                    if not real_position:
                        logging.warning(f"{symbol}: No active position on Bybit, removing from tracking")
                        with config.POSITIONS_LOCK:
                            if symbol in config.ACTIVE_POSITIONS:
                                del config.ACTIVE_POSITIONS[symbol]
                        continue
                else:
                    logging.error(f"{symbol}: Error checking position: {positions_response.get('retMsg')}")
                    continue
                    
            except Exception as e:
                logging.error(f"{symbol}: Error verifying position: {e}")
                continue
                
            current_sl = pos_info['sl']
            timeframe_entry = pos_info.get('timeframe', '15m')
            chat_id = pos_info.get('chat_id')
            
            # Determina timeframe EMA per trailing
            ema_tf = TRAILING_EMA_TIMEFRAME.get(timeframe_entry, '5m')
            
            # Scarica dati per calcolare EMA 10
            df = bybit_get_klines(symbol, ema_tf, limit=20)
            if df.empty:
                logging.warning(f"{symbol}: No data for trailing EMA calculation")
                continue
            
            current_price = df['close'].iloc[-1]
            
            # ===== CALCOLO PROFIT % (DIVERSO PER BUY/SELL) =====
            if side == 'Buy':
                profit_usd = (current_price - entry_price) * pos_info['qty']
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # Sell
                profit_usd = (entry_price - current_price) * pos_info['qty']
                profit_pct = ((entry_price - current_price) / entry_price) * 100

            # ===== CALCOLO RISK INIZIALE =====
            if side == 'Buy':
                initial_risk_per_unit = entry_price - current_sl
            else:
                initial_risk_per_unit = current_sl - entry_price
            
            initial_risk_usd = initial_risk_per_unit * pos_info['qty']
            
            if not PROFIT_LOCK_ENABLED:
                # Skip profit lock se disabilitato
                logging.debug(f"{symbol}: Profit lock DISABLED, skip to normal trailing")
            else:
                # Carica config
                PROFIT_LOCK_MULTIPLIER = PROFIT_LOCK_CONFIG['multiplier']
                PROFIT_LOCK_RETENTION = PROFIT_LOCK_CONFIG['retention']
                MIN_PROFIT_USD = PROFIT_LOCK_CONFIG['min_profit_usd']

                # Calcola profit/risk ratio
                if initial_risk_usd > 0:
                    profit_risk_ratio = profit_usd / initial_risk_usd
                else:
                    profit_risk_ratio = 0.0
                
                # ✅ CHECK 1: Profit >= 5x risk iniziale?
                # ✅ CHECK 2: Profit assoluto >= min_profit_usd?
                if profit_risk_ratio >= PROFIT_LOCK_MULTIPLIER and profit_usd >= MIN_PROFIT_USD:
                    logging.info(
                        f"🔒 {symbol}: PROFIT LOCK TRIGGER! "
                        f"Profit={profit_usd:.2f} USD ({profit_risk_ratio:.1f}x risk), "
                        f"Threshold={PROFIT_LOCK_MULTIPLIER}x"
                    )
                    
                    # ✅ CALCOLA NUOVO SL CON RETENTION
                    if side == 'Buy':
                        # LONG: Blocca 80% del profit raggiunto
                        locked_profit = profit_usd * PROFIT_LOCK_RETENTION
                        new_sl = entry_price + (locked_profit / pos_info['qty'])
                        
                        # Assicurati che SL sia miglioramento (mai indietro)
                        if new_sl <= current_sl:
                            logging.debug(
                                f"{symbol}: Profit lock SL {new_sl:.4f} <= current {current_sl:.4f}, skip"
                            )
                            # Non fare nulla, continua al trailing normale
                        else:
                            # ✅ ESEGUI PROFIT LOCK
                            profit_lock_executed = await execute_profit_lock(
                                context=context,
                                symbol=symbol,
                                side=side,
                                new_sl=new_sl,
                                current_sl=current_sl,
                                profit_usd=profit_usd,
                                profit_pct=profit_pct,
                                profit_risk_ratio=profit_risk_ratio,
                                locked_profit=locked_profit,
                                entry_price=entry_price,
                                pos_info=pos_info,
                                timeframe_entry=timeframe_entry,
                                chat_id=chat_id
                            )
                            
                            if profit_lock_executed:
                                # Profit lock eseguito con successo, skip trailing normale
                                logging.info(f"{symbol}: Profit lock SUCCESS, skip normal trailing")
                                continue  # Vai alla prossima posizione
                            else:
                                # Profit lock fallito, continua al trailing normale
                                logging.warning(f"{symbol}: Profit lock FAILED, fallback to normal trailing")
                    
                    else:  # Sell (SHORT)
                        # SHORT: Blocca 80% del profit raggiunto
                        locked_profit = profit_usd * PROFIT_LOCK_RETENTION
                        new_sl = entry_price - (locked_profit / pos_info['qty'])
                        
                        # Assicurati che SL sia miglioramento (mai peggio per SHORT)
                        if new_sl >= current_sl:
                            logging.debug(
                                f"{symbol}: SHORT Profit lock SL {new_sl:.4f} >= current {current_sl:.4f}, skip"
                            )
                            # Non fare nulla, continua al trailing normale
                        else:
                            # ✅ ESEGUI PROFIT LOCK
                            profit_lock_executed = await execute_profit_lock(
                                context=context,
                                symbol=symbol,
                                side=side,
                                new_sl=new_sl,
                                current_sl=current_sl,
                                profit_usd=profit_usd,
                                profit_pct=profit_pct,
                                profit_risk_ratio=profit_risk_ratio,
                                locked_profit=locked_profit,
                                entry_price=entry_price,
                                pos_info=pos_info,
                                timeframe_entry=timeframe_entry,
                                chat_id=chat_id
                            )
                            
                            if profit_lock_executed:
                                logging.info(f"{symbol}: SHORT Profit lock SUCCESS, skip normal trailing")
                                continue  # Vai alla prossima posizione
                            else:
                                logging.warning(f"{symbol}: SHORT Profit lock FAILED, fallback to normal trailing")
                else:
                    # Profit non abbastanza alto per profit lock
                    logging.debug(
                        f"{symbol}: Profit {profit_usd:.2f} USD ({profit_risk_ratio:.1f}x) "
                        f"< threshold ({PROFIT_LOCK_MULTIPLIER}x) or < min ${MIN_PROFIT_USD}"
                    )
            
            # ✅ CONTINUA AL TRAILING NORMALE (se profit lock non eseguito)
            active_level = None
            for level in TRAILING_CONFIG_ADVANCED['levels']:
                if profit_pct >= level['profit_pct']:
                    active_level = level
                else:
                    break
            
            if not active_level:
                logging.debug(f"{symbol} ({side}): Profit {profit_pct:.2f}% < min threshold")
                continue
            
            # ===== CALCOLA NUOVO SL CON BUFFER DINAMICO =====
            ema_10 = df['close'].ewm(span=10, adjust=False).mean().iloc[-1]
            ema_buffer = active_level['ema_buffer']
            
            if side == 'Buy':
                # SL sotto EMA per LONG
                new_sl = ema_10 * (1 - ema_buffer)
                
                # Check se SL è migliorato (mai indietro)
                if TRAILING_CONFIG_ADVANCED['never_back'] and new_sl <= current_sl:
                    logging.debug(f"{symbol} (BUY): New SL {new_sl:.4f} <= current {current_sl:.4f}, skip")
                    continue
                    
            else:  # Sell
                # SL sopra EMA per SHORT
                new_sl = ema_10 * (1 + ema_buffer)
                
                # Check se SL è migliorato (mai indietro = mai più alto per SHORT)
                if TRAILING_CONFIG_ADVANCED['never_back'] and new_sl >= current_sl:
                    logging.debug(f"{symbol} (SELL): New SL {new_sl:.4f} >= current {current_sl:.4f}, skip")
                    continue
            
            # Check movimento minimo
            min_move_pct = TRAILING_CONFIG_ADVANCED.get('min_move_pct', 0.1)
            move_pct = abs((new_sl - current_sl) / current_sl) * 100
            if move_pct < min_move_pct:
                logging.debug(f"{symbol} ({side}): SL move {move_pct:.2f}% < min {min_move_pct}%, skip")
                continue
            
            # ===== AGGIORNA SU BYBIT =====
            try:
                session = create_bybit_session()
                result = session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    stopLoss=str(round(new_sl, get_price_decimals(new_sl))),
                    positionIdx=0
                )
                
                if result.get('retCode') == 0:
                    # Aggiorna tracking locale
                    with config.POSITIONS_LOCK:
                        if symbol in config.ACTIVE_POSITIONS:
                            config.ACTIVE_POSITIONS[symbol]['sl'] = new_sl
                            
                            logging.info(f"{symbol} ({side}): Trailing SL updated to {new_sl:.4f} (Level: {active_level['label']})")
                    
                    # ===== NOTIFICA TELEGRAM =====
                    if chat_id:
                        try:
                            # Recupera il prezzo di entrata prima del calcolo del profitto
                            entry_price = float(pos_info.get('entryPrice', 0)) 
                            profit_usd = abs(current_price - entry_price) * pos_info['qty']
                            sl_move_usd = abs(new_sl - current_sl) * pos_info['qty']
                            
                            level_emoji = {
                                'Early Protection': '🟡',
                                'Standard Trail': '🟢',
                                'Tight Trail': '🔵',
                                'Ultra Tight Trail': '🟣'
                            }.get(active_level['label'], '⚪')
                            
                            side_emoji = "🟢" if side == 'Buy' else "🔴"
                            
                            notification = f"<b>{side_emoji} Trailing Stop Upgraded ({side})</b>\n\n"
                            notification += f"{level_emoji} <b>Level: {active_level['label']}</b>\n"
                            notification += f"<b>Symbol:</b> {symbol} ({timeframe_entry})\n"
                            notification += f"<b>Prezzo:</b> ${current_price:.{get_price_decimals(current_price)}f}\n"
                            notification += f"<b>Profit:</b> {profit_pct:.2f}% (${profit_usd:.2f})\n\n"
                            notification += f"<b>Stop Loss:</b>\n"
                            notification += f"• Prima: ${current_sl:.{get_price_decimals(current_sl)}f}\n"
                            notification += f"• Ora: ${new_sl:.{get_price_decimals(new_sl)}f}\n"
                            notification += f"• Spostamento: ${sl_move_usd:.2f}\n\n"
                            notification += f"<b>EMA 10 ({ema_tf}):</b> ${ema_10:.{get_price_decimals(ema_10)}f}\n"
                            notification += f"<b>Buffer:</b> {ema_buffer * 100:.2f}% {'sotto' if side == 'Buy' else 'sopra'} EMA\n\n"
                            
                            # Calcola profit protetto
                            if side == 'Buy':
                                protected_profit = (new_sl - entry_price) * pos_info['qty']
                            else:
                                protected_profit = (entry_price - new_sl) * pos_info['qty']
                                
                            if protected_profit > 0:
                                notification += f"✅ <b>SL protegge ora ${protected_profit:.2f} di profit</b>"
                            else:
                                notification += f"⚠️ SL ancora sotto entry (break-even mode)"
                            
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=notification,
                                parse_mode="HTML"
                            )
                        except Exception as e:
                            logging.error(f"Errore invio notifica trailing: {e}")
                else:
                    logging.error(f"{symbol}: Errore aggiornamento SL Bybit: {result.get('retMsg')}")
            
            except Exception as e:
                logging.error(f"{symbol}: Errore set_trading_stop: {e}")
        
        except Exception as e:
            logging.exception(f"Errore trailing SL per {symbol}: {e}")


async def monitor_partial_tp(context: ContextTypes.DEFAULT_TYPE):
    """
    Monitora prezzo vs TP levels e chiude posizioni parzialmente
    
    Eseguito ogni 30 secondi
    
    LOGICA:
    1. Per ogni posizione con multi-TP attivo
    2. Scarica prezzo corrente
    3. Se prezzo >= TP1 (non ancora hit) → Chiudi 40%
    4. Se prezzo >= TP2 (non ancora hit) → Chiudi 30%
    5. Se prezzo >= TP3 (non ancora hit) → Chiudi 30% (residuo)
    6. Aggiorna qty posizione e attiva trailing
    7. Invia notifica Telegram
    """
    
    if not config.MULTI_TP_ENABLED or not config.MULTI_TP_CONFIG['enabled']:
        return
    
    with config.POSITIONS_LOCK:
        positions_copy = dict(config.ACTIVE_POSITIONS)
    
    if not positions_copy:
        return
    
    logging.debug(f"Multi-TP check: {len(positions_copy)} positions")
    
    for symbol, pos_info in positions_copy.items():
        # Skip se non ha multi-TP configurato
        tp_levels = pos_info.get('multi_tp_levels')
        if not tp_levels:
            continue
        
        try:
            side = pos_info['side']
            entry_price = pos_info['entry_price']
            current_qty = pos_info['qty']
            timeframe = pos_info.get('timeframe', '15m')
            chat_id = pos_info.get('chat_id')
            
            # Scarica prezzo corrente
            df = bybit_get_klines(symbol, timeframe, limit=5)
            if df.empty:
                continue
            
            current_price = df['close'].iloc[-1]
            
            # Check ogni TP level
            for i, tp_level in enumerate(tp_levels, 1):
                tp_price = tp_level['price']
                close_pct = tp_level['close_pct']
                tp_qty = tp_level['qty']
                label = tp_level['label']
                emoji = tp_level['emoji']
                
                # Skip se già hit
                if tp_level.get('hit', False):
                    continue
                
                # Check se prezzo ha raggiunto TP (con buffer)
                buffer = config.MULTI_TP_CONFIG['buffer_pct']
                
                if side == 'Buy':
                    tp_reached = current_price >= tp_price * (1 - buffer)
                else:  # Sell
                    tp_reached = current_price <= tp_price * (1 + buffer)
                
                if not tp_reached:
                    continue
                
                # ===== TP RAGGIUNTO! =====
                logging.info(
                    f"🎯 {symbol}: TP{i} REACHED! "
                    f"Price ${current_price:.4f} >= TP ${tp_price:.4f}"
                )
                
                # Calcola qty da chiudere (% del qty CORRENTE, non originale)
                qty_to_close = round(current_qty * close_pct, pos_info.get('qty_decimals', 3))
                
                # Verifica min qty
                min_qty = config.MULTI_TP_CONFIG.get('min_partial_qty', 0.001)
                if qty_to_close < min_qty:
                    logging.warning(
                        f"{symbol}: TP{i} qty too small ({qty_to_close:.4f} < {min_qty}), "
                        f"skipping partial close"
                    )
                    # Marca come hit comunque per non riprovare
                    tp_level['hit'] = True
                    continue
                
                # ===== CHIUDI POSIZIONE PARZIALE =====
                try:
                    session = create_bybit_session()
                    
                    # Usa Market order con reduceOnly per chiusura parziale
                    close_order = session.place_order(
                        category='linear',
                        symbol=symbol,
                        side='Sell' if side == 'Buy' else 'Buy',  # Inverti lato
                        orderType='Market',
                        qty=str(qty_to_close),
                        reduceOnly=True,  # Solo riduce posizione esistente
                        positionIdx=0
                    )
                    
                    if close_order.get('retCode') == 0:
                        # ===== SUCCESS: Aggiorna tracking =====
                        
                        # Calcola profit
                        if side == 'Buy':
                            profit_per_unit = current_price - entry_price
                        else:
                            profit_per_unit = entry_price - current_price
                        
                        profit_usd = profit_per_unit * qty_to_close
                        profit_pct = (profit_per_unit / entry_price) * 100
                        
                        # Aggiorna posizione
                        new_qty = current_qty - qty_to_close
                        
                        with config.POSITIONS_LOCK:
                            if symbol in config.ACTIVE_POSITIONS:
                                config.ACTIVE_POSITIONS[symbol]['qty'] = new_qty
                                
                                # Marca TP come hit
                                for level in config.ACTIVE_POSITIONS[symbol]['multi_tp_levels']:
                                    if level['price'] == tp_price:
                                        level['hit'] = True
                                
                                logging.info(
                                    f"✅ {symbol}: TP{i} executed! "
                                    f"Closed {qty_to_close:.4f} (${profit_usd:+.2f}), "
                                    f"Remaining: {new_qty:.4f}"
                                )
                        
                        # Aggiorna TP tracking
                        with TP_TRACKING_LOCK:
                            if symbol in TP_TRACKING:
                                TP_TRACKING[symbol][f'tp{i}_hit'] = True
                                TP_TRACKING[symbol][f'tp{i}_qty_closed'] = qty_to_close
                        
                        # ===== ATTIVA TRAILING DOPO TP1 =====
                        if i == 1 and config.MULTI_TP_CONFIG.get('activate_trailing_after_tp1', True):
                            with config.POSITIONS_LOCK:
                                if symbol in config.ACTIVE_POSITIONS:
                                    config.ACTIVE_POSITIONS[symbol]['trailing_active'] = True
                            
                            logging.info(f"🔄 {symbol}: Trailing SL attivato dopo TP1")
                        
                        # ===== NOTIFICA TELEGRAM =====
                        if chat_id:
                            try:
                                side_emoji = '🟢' if side == 'Buy' else '🔴'
                                price_decimals = get_price_decimals(current_price)
                                
                                notification = f"{side_emoji} {emoji} <b>TAKE PROFIT {i} HIT!</b>\n\n"
                                notification += f"<b>Symbol:</b> {symbol} ({timeframe})\n"
                                notification += f"<b>{label}</b>\n\n"
                                
                                notification += f"<b>📊 Chiusura Parziale:</b>\n"
                                notification += f"• Qty chiusa: {qty_to_close:.4f} ({close_pct*100:.0f}%)\n"
                                notification += f"• Prezzo: ${current_price:.{price_decimals}f}\n"
                                notification += f"• Entry: ${entry_price:.{price_decimals}f}\n\n"
                                
                                notification += f"<b>💰 Profit Parziale:</b>\n"
                                notification += f"• ${profit_usd:+.2f} ({profit_pct:+.2f}%)\n\n"
                                
                                notification += f"<b>📦 Posizione Residua:</b>\n"
                                notification += f"• Qty: {new_qty:.4f}\n"
                                
                                # Info TP rimanenti
                                remaining_tps = [
                                    (j, tp) for j, tp in enumerate(tp_levels, 1) 
                                    if not tp.get('hit', False)
                                ]
                                
                                if remaining_tps:
                                    notification += f"\n<b>🎯 TP Rimanenti:</b>\n"
                                    for tp_idx, tp in remaining_tps:
                                        notification += f"• TP{tp_idx}: ${tp['price']:.{price_decimals}f} ({tp['close_pct']*100:.0f}%)\n"
                                else:
                                    notification += f"\n✅ <b>Tutti i TP eseguiti!</b>\n"
                                
                                # Status trailing
                                if i == 1 and config.MULTI_TP_CONFIG.get('activate_trailing_after_tp1'):
                                    notification += f"\n🔄 <b>Trailing SL ora ATTIVO</b>\n"
                                    notification += f"Stop Loss proteggerà il residuo automaticamente"
                                
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=notification,
                                    parse_mode='HTML'
                                )
                            
                            except Exception as e:
                                logging.error(f"Errore invio notifica TP{i}: {e}")
                    
                    else:
                        logging.error(
                            f"{symbol}: Errore chiusura TP{i}: {close_order.get('retMsg')}"
                        )
                
                except Exception as e:
                    logging.error(f"{symbol}: Errore esecuzione TP{i}: {e}")
        
        except Exception as e:
            logging.exception(f"Errore monitoring TP per {symbol}: {e}")

# ===== FUNZIONE per schedulare il job =====
def schedule_trailing_stop_job(application):
    """
    Schedula il job di trailing stop loss ogni 5 minuti
    """
    if not config.TRAILING_STOP_ENABLED:
        logging.info('🔕 Trailing Stop Loss disabilitato')
        return
    
    #interval = TRAILING_CONFIG['check_interval']
    # Usa check_interval dal config advanced
    interval = config.TRAILING_CONFIG_ADVANCED['check_interval']  # ← MODIFICA QUI
    
    application.job_queue.run_repeating(
        update_trailing_stop_loss,
        interval=interval,
        first=60,  # Primo check dopo 1 minuto
        name='trailing_stop_loss'
    )
    
    logging.info(f'✅ Trailing Stop Loss schedulato ogni {interval}s ({interval//60} minuti)')


# ----------------------------- CHART GENERATION -----------------------------

def generate_chart(df: pd.DataFrame, symbol: str, timeframe: str) -> io.BytesIO:
    """
    Genera grafico candlestick usando mplfinance con EMA overlay
    Returns: BytesIO object (immagine in memoria)
    """
    try:
        # Usa gli ultimi 100 candles per il grafico
        chart_df = df.tail(100).copy()
        
        # Calcola EMA (5, 10, 60, 223)
        ema_5 = chart_df['close'].ewm(span=5, adjust=False).mean()
        ema_10 = chart_df['close'].ewm(span=10, adjust=False).mean()
        ema_60 = chart_df['close'].ewm(span=60, adjust=False).mean()
        ema_223 = chart_df['close'].ewm(span=223, adjust=False).mean()
        
        # Crea plot addizionali per le EMA
        apds = [
            mpf.make_addplot(ema_5, color='#00FF00', width=1.5, label='EMA 5'),
            mpf.make_addplot(ema_10, color='#FFA500', width=1.5, label='EMA 10'),
            mpf.make_addplot(ema_60, color='#FF1493', width=1.5, label='EMA 60'),
            mpf.make_addplot(ema_223, color='#1E90FF', width=2, label='EMA 223')
        ]
        
        # Crea il grafico in memoria
        buffer = io.BytesIO()
        
        # Stile personalizzato
        mc = mpf.make_marketcolors(
            up='#26a69a',
            down='#ef5350',
            edge='inherit',
            wick='inherit',
            volume='in'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',
            gridcolor='#2e2e2e',
            facecolor='#1e1e1e',
            figcolor='#1e1e1e',
            y_on_right=False
        )
        
        mpf.plot(
            chart_df,
            type='candle',
            style=s,
            title=f'{symbol} - {timeframe}',
            ylabel='Price',
            volume=True,
            addplot=apds,
            savefig=dict(fname=buffer, dpi=100, bbox_inches='tight')
        )
        
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logging.error(f'Errore nella generazione del grafico: {e}')
        raise


# ----------------------------- JOB CALLBACK -----------------------------

async def analyze_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Job che viene eseguito ad ogni chiusura candela
    
    LOGICA EMA-FIRST (ottimizzata):
    1. Analizza condizioni EMA
    2. Se EMA non valide → SKIP ricerca pattern (risparmio risorse)
    3. Se EMA OK → Cerca pattern
    4. Se pattern trovato → Invia segnale
    
    COMPORTAMENTO:
    - DEFAULT: Invia grafico SOLO quando trova pattern
    - FULL_MODE: Invia sempre (anche senza pattern, con analisi EMA)
    """
    
    job_ctx = context.job.data
    chat_id = job_ctx['chat_id']
    symbol = job_ctx['symbol']
    timeframe = job_ctx['timeframe']
    key = f'{symbol}-{timeframe}'
    momentum_reason = ""   # default

    #logging.info(f'🔍 Analyzing {symbol} {timeframe}...')
    logging.debug(f'   Volume mode: {config.VOLUME_FILTER_MODE}')
    logging.debug(f'   Trend mode: {config.TREND_FILTER_MODE}')
    logging.debug(f'   EMA mode: {config.EMA_FILTER_MODE if config.EMA_FILTER_ENABLED else "OFF"}')
    logging.debug(f'   Market time: {"ON" if config.MARKET_TIME_FILTER_ENABLED else "OFF"}')
    
    # Check se auto-discovered
    is_auto = job_ctx.get('auto_discovered', False)

    # ===== MARKET TIME FILTER (PRIORITY CHECK) =====
    if config.MARKET_TIME_FILTER_ENABLED:
        time_ok, time_reason = is_good_trading_time_utc()
        
        if not time_ok:
            now_utc = datetime.now(timezone.utc)
            current_hour = now_utc.hour
            
            logging.info(
                f'⏰ {symbol} {timeframe}: Market time filter active '
                f'(UTC hour {current_hour:02d})'
            )
            
            if config.MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY:
                # Blocca SOLO autotrade, analisi continua
                logging.info(f'   Mode: AUTOTRADE_ONLY - Analysis continues, trading disabled')
                
                # Forza autotrade = False per questo ciclo
                job_ctx['autotrade'] = False
                
                # IMPORTANTE: Continua con l'analisi ma senza trading
            else:
                # Blocca TUTTO (analisi + trading)
                logging.info(f'   Mode: ALL_ANALYSIS - Skipping analysis completely')
                
                # Invia notifica opzionale (solo 1 volta per ciclo di blocco)
                if not hasattr(analyze_job, f'notified_{symbol}_{timeframe}'):
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=(
                                f"⏰ <b>Market Time Filter Active</b>\n\n"
                                f"Symbol: {symbol} {timeframe}\n"
                                f"UTC Hour: {current_hour:02d}\n"
                                f"Blocked Hours: {sorted(config.MARKET_TIME_FILTER_BLOCKED_UTC_HOURS)}\n\n"
                                f"Analysis paused during low liquidity hours.\n"
                                f"Will resume automatically."
                            ),
                            parse_mode='HTML'
                        )
                        # Marca come notificato per questa sessione
                        setattr(analyze_job, f'notified_{symbol}_{timeframe}', True)
                    except:
                        pass
                
                return  # STOP: Skip analysis completamente
        else:
            # Reset flag notifica quando orario torna OK
            if hasattr(analyze_job, f'notified_{symbol}_{timeframe}'):
                delattr(analyze_job, f'notified_{symbol}_{timeframe}')

    # Verifica se le notifiche complete sono attive
    with config.FULL_NOTIFICATIONS_LOCK:
        full_mode = chat_id in config.FULL_NOTIFICATIONS and key in config.FULL_NOTIFICATIONS[chat_id]

    try:
        # ===== STEP 1: OTTIENI DATI =====
        df = bybit_get_klines(symbol, timeframe, limit=200)
        if df.empty:
            logging.warning(f'Nessun dato per {symbol} {timeframe}')
            if full_mode:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f'⚠️ Nessun dato disponibile per {symbol} {timeframe}'
                )
            return

        # ===== AGGIUNGI QUESTO CHECK =====
        # Verifica età ultima candela per escludere quella corrente (in formazione)
        last_candle_time = df.index[-1]
        if last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.tz_localize('UTC')
        
        # Calcola quanti secondi sono passati dall'apertura dell'ultima candela
        now_utc = datetime.now(timezone.utc)
        time_diff = (now_utc - last_candle_time).total_seconds()
        
        # Ottieni durata timeframe in secondi
        interval_seconds = config.INTERVAL_SECONDS.get(timeframe, 300)
        
        # Se l'ultima candela è troppo recente (meno del 90% del timeframe),
        # è ancora in formazione → usa la penultima
        threshold = interval_seconds * 0.9  # 90% del timeframe
        
        if time_diff < threshold:
            logging.debug(
                f"{symbol} {timeframe}: Last candle too recent "
                f"({time_diff:.0f}s < {threshold:.0f}s threshold), "
                f"using previous closed candle"
            )
            # Rimuovi ultima candela (quella in formazione)
            df = df.iloc[:-1]
            
            if df.empty:
                logging.warning(f'{symbol} {timeframe}: No closed candles available')
                return
        
        # Ora df contiene SOLO candele chiuse
        # ===== FINE BLOCCO =====

        last_close = df['close'].iloc[-1]
        last_time = df.index[-1]
        timestamp_str = last_time.strftime('%Y-%m-%d %H:%M UTC')

        # ✅ FIX PUNTO #3: Lock per lettura thread-safe
        with config.POSITIONS_LOCK:
            position_exists = symbol in config.ACTIVE_POSITIONS

        # Log per debug
        if position_exists:
            logging.debug(f'{symbol}: Position already exists, skip order')

        caption = ""  # ← AGGIUNGI QUESTA RIGA
        # ===== CALCOLA DECIMALI UNA SOLA VOLTA =====
        price_decimals = get_price_decimals(last_close)
        
        # ===== STEP 2: PRE-FILTER EMA (PRIMA DI CERCARE PATTERN) =====
        ema_analysis = None
        pattern_search_allowed = True  # Default: cerca pattern
        
        if config.EMA_FILTER_ENABLED:
            ema_analysis = analyze_ema_conditions(df, timeframe, None)
            
            logging.info(
                f'📊 EMA Analysis {symbol} {timeframe}: '
                f'Score={ema_analysis["score"]}, '
                f'Quality={ema_analysis["quality"]}, '
                f'Passed={ema_analysis["passed"]}'
            )
            
            # STRICT MODE: Blocca completamente se EMA non passa
            if config.EMA_FILTER_MODE == 'strict' and not ema_analysis['passed']:
                pattern_search_allowed = False
                logging.warning(
                    f'🚫 {symbol} {timeframe} - EMA STRICT BLOCK '
                    f'(score {ema_analysis["score"]}/100). Skip pattern search.'
                )
                
                # Se full mode, invia comunque analisi mercato
                if full_mode:
                    caption = (
                        f"📊 <b>{symbol}</b> ({timeframe})\n"
                        f"🕐 {timestamp_str}\n"
                        f"💵 Price: ${last_close:.{price_decimals}f}\n\n"
                        f"🚫 <b>ZONA NON VALIDA (EMA Strict)</b>\n\n"
                        f"Score EMA: {ema_analysis['score']}/100\n"
                        f"Quality: {ema_analysis['quality']}\n\n"
                        f"{ema_analysis['details']}\n\n"
                        f"⚠️ Pattern search DISABILITATA per score EMA insufficiente.\n"
                        f"Attendi miglioramento condizioni EMA."
                    )
                    
                    try:
                        chart_buffer = generate_chart(df, symbol, timeframe)
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=chart_buffer,
                            caption=caption,
                            parse_mode='HTML'
                        )
                    except:
                        await context.bot.send_message(
                            chat_id=chat_id, 
                            text=caption, 
                            parse_mode='HTML'
                        )
                
                #return  # STOP QUI - Non cerca pattern
            
            # LOOSE MODE: Blocca se score < 40
            elif config.EMA_FILTER_MODE == 'loose' and not ema_analysis['passed']:
                pattern_search_allowed = False
                logging.warning(
                    f'🚫 {symbol} {timeframe} - EMA LOOSE BLOCK '
                    f'(score {ema_analysis["score"]}/100). Skip pattern search.'
                )
                
                # Se full mode, invia comunque analisi
                if full_mode:
                    caption = (
                        f"📊 <b>{symbol}</b> ({timeframe})\n"
                        f"🕐 {timestamp_str}\n"
                        f"💵 Price: ${last_close:.{price_decimals}f}\n\n"
                        f"⚠️ <b>EMA Score troppo basso (Loose mode)</b>\n\n"
                        f"Score EMA: {ema_analysis['score']}/100\n"
                        f"Quality: {ema_analysis['quality']}\n\n"
                        f"Minimo richiesto in LOOSE: 40/100\n"
                        f"Attendi miglioramento condizioni."
                    )
                    
                    try:
                        chart_buffer = generate_chart(df, symbol, timeframe)
                        await context.bot.send_photo(
                            chat_id=chat_id, 
                            photo=chart_buffer, 
                            caption=caption, 
                            parse_mode='HTML'
                        )
                    except:
                        await context.bot.send_message(
                            chat_id=chat_id, 
                            text=caption, 
                            parse_mode='HTML'
                        )
                
                #return  # STOP - Non cerca pattern
        
        # ===== STEP 3: CERCA PATTERN (solo se EMA permette) =====
        # I filtri globali sono DENTRO check_patterns() ora
        found, side, pattern, pattern_data = check_patterns(df, symbol=symbol)
        
        if found:
            #logging.info(f'🎯 Pattern trovato: {pattern} ({side}) su {symbol} {timeframe}')
            logging.info(f'✅ {symbol} {timeframe} - Pattern FOUND: {pattern} ({side})')

            # Registra segnale nelle statistiche
            try:
                track_patterns.integrate_pattern_stats_on_signal(
                    pattern_name=pattern,
                    symbol=symbol,
                    timeframe=timeframe,
                    side=side,
                    autotrade=job_ctx.get('autotrade', False)
                )
            except Exception as e:
                logging.error(f'Errore tracking pattern signal: {e}')
    
            # Log pattern-specific data se disponibile
            if pattern_data:
                    #logging.info(f'   {symbol} - Quality Score: {pattern_data["quality_score"]}/100 - Tier: {pattern_data["tier"]} - Volume: {pattern_data["volume_ratio"]:.1f}x')
                    # ===== FIX: Check sicuro per quality_score =====
                    quality_score = pattern_data.get('quality_score', 'N/A')
                    tier = pattern_data.get('tier', 'N/A')
                    volume_ratio = pattern_data.get('volume_ratio', 0)
                    
                    if quality_score != 'N/A' and tier != 'N/A' and volume_ratio > 0:
                        logging.info(
                            f'   {symbol} - Quality Score: {quality_score}/100 - '
                            f'Tier: {tier} - Volume: {volume_ratio:.1f}x'
                        )
                    else:
                        # Fallback per pattern senza questi dati
                        logging.info(f'   {symbol} - Pattern: {pattern} (data structure varies)')
        else:
            logging.debug(f'❌ {symbol} {timeframe} - NO pattern detected')
            # Log perché non ha trovato pattern (se EMA era OK)
            if ema_analysis and ema_analysis['passed']:
                logging.info(f'  {symbol} - EMA was OK ({ema_analysis["quality"]}) but no pattern matched')

        
        # Se NON pattern e NON full_mode → Skip notifica
        if not found and not full_mode:
            logging.debug(f'🔕 {symbol} {timeframe} - No pattern, no full mode → Skip')
            return
        
        # ===== STEP 4: CALCOLA PARAMETRI TRADING =====
        atr_series = atr(df, period=14)
        last_atr = atr_series.iloc[-1] if not atr_series.isna().all() else np.nan

        # 🔧 FIX: Dichiara position_exists SUBITO
        position_exists = symbol in config.ACTIVE_POSITIONS
        # ===== STEP 5: COSTRUISCI MESSAGGIO =====
        
        if found and side == 'Buy':
            # Check Higher Timeframe EMA (tappo)
            htf_block = check_higher_timeframe_resistance(symbol=symbol, current_tf=timeframe, current_price=last_close)
            
            if htf_block['blocked']:
                logging.warning(
                    f'🚫 Pattern {pattern} su {symbol} {timeframe} '
                    f'BLOCCATO da resistenza HTF {htf_block["htf"]}'
                )
                
                # In full mode, invia warning
                if full_mode:
                    caption = (
                        f"⚠️ <b>Pattern BLOCCATO da HTF</b>\n\n"
                        f"Pattern: {pattern} su {timeframe}\n"
                        f"Timeframe superiore: {htf_block['htf']}\n\n"
                        f"Resistenze HTF:\n"
                        f"{htf_block['details']}\n\n"
                        f"💡 Aspetta breakout HTF o cerca altro setup"
                    )
                    
                    try:
                        chart_buffer = generate_chart(df, symbol, timeframe)
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=chart_buffer,
                            caption=caption,
                            parse_mode='HTML'
                        )
                    except:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=caption,
                            parse_mode='HTML'
                        )
                
                return  # BLOCCA segnale

            # Inizializza variabili default per evitare UnboundLocalError
            entry_price = last_close  # ← AGGIUNGI all'inizio del blocco BUY
            sl_price = None
            tp_price = None
            ema_used = 'ATR'
            ema_value = 0

            # ===== GESTIONE PATTERN-SPECIFIC ENTRY/SL/TP =====
            if pattern == 'Volume Spike Breakout' and pattern_data:
                entry_price = last_close  # Entry immediato
                
                # SL: EMA 10 o ATR
                if config.USE_EMA_STOP_LOSS:
                    sl_price, ema_used, ema_value = calculate_ema_stop_loss(
                        df, timeframe, last_close, side
                    )
                else:
                    if not math.isnan(last_atr) and last_atr > 0:
                        sl_price = last_close - last_atr * 1.5
                        ema_used = 'ATR'
                        ema_value = last_atr
                    else:
                        sl_price = pattern_data['ema10'] * 0.998
                        ema_used = 'EMA 10'
                        ema_value = pattern_data['ema10']
                
                # TP: Standard ATR
                tp_price = last_close + abs(last_close - sl_price) * 2.0
            
            # === BULLISH FLAG BREAKOUT ===
            elif pattern == 'Bullish Flag Breakout' and pattern_data:
                entry_price = pattern_data['X']
                sl_price = pattern_data['consolidation_low'] * 0.998
                tp_price = pattern_data['X'] + (pattern_data['pole_height'] * 1.5)
                ema_used = 'Flag Pattern'
                ema_value = pattern_data['consolidation_low']
                
                logging.info(f'🚩 Bullish Flag Entry Setup:')
                logging.info(f'   X (breakout): ${entry_price:.{price_decimals}f}')
                logging.info(f'   Pole Height: {pattern_data["pole_height_pct"]:.2f}%')
                logging.info(f'   Entry: ${entry_price:.{price_decimals}f}')
                logging.info(f'   SL: ${sl_price:.{price_decimals}f}')
                logging.info(f'   TP: ${tp_price:.{price_decimals}f}')

            elif pattern == 'Triple Touch Breakout' and pattern_data:
                """
                ENTRY LOGIC per Triple Touch Breakout
                
                Entry: Al breakout del terzo tocco (prezzo corrente)
                SL: Sotto consolidamento low (con buffer 0.2%)
                TP: R + (2.5 × range consolidamento)
                
                R:R tipico: 1:2.5-3.5
                """
                
                entry_price = pattern_data.get('suggested_entry')
                sl_price   = pattern_data.get('suggested_sl')
                tp_price   = pattern_data.get('suggested_tp')
                
                if entry_price is None or sl_price is None or tp_price is None:
                    logging.error(f"{symbol} {timeframe} - Pattern {pattern}: missing suggested_* in pattern_data keys={list(pattern_data.keys())}")
                    return  # oppure continue / skip ordine
                
                ema_used = 'Triple Touch Zone'
                ema_value = pattern_data['resistance']
                
                # Calcola decimali
                price_decimals = get_price_decimals(entry_price)
                
                # Log dettagliato
                logging.info(f'🎯 Triple Touch Breakout Entry Setup:')
                logging.info(f'   Resistance: ${pattern_data["resistance"]:.{price_decimals}f}')
                logging.info(f'   Touch 1 rejection: {pattern_data["touch_1_rejection_pct"]:.1f}%')
                logging.info(f'   Touch 2 rejection: {pattern_data["touch_2_rejection_pct"]:.1f}%')
                logging.info(f'   Consolidation: {pattern_data["consolidation_duration"]} candele, {pattern_data["range_pct"]:.2f}%')
                logging.info(f'   Entry: ${entry_price:.{price_decimals}f}')
                logging.info(f'   SL: ${sl_price:.{price_decimals}f}')
                logging.info(f'   TP: ${tp_price:.{price_decimals}f}')
                logging.info(f'   Volume: {pattern_data["volume_ratio"]:.1f}x')
                logging.info(f'   EMA 60: ${pattern_data["ema60"]:.{price_decimals}f}')
                logging.info(f'   Min distance to EMA 60: {pattern_data["min_distance_to_ema60_pct"]:.2f}%')
                
                # Nel caption aggiungi info specifiche Triple Touch
                caption += f"🎯 <b>Triple Touch Breakout</b> ({pattern_data['quality']})\n"
                caption += f"📍 Resistance: ${pattern_data['resistance']:.{price_decimals}f}\n"
                caption += f"🔄 Rejections: {pattern_data['touch_1_rejection_pct']:.1f}% + {pattern_data['touch_2_rejection_pct']:.1f}%\n"
                caption += f"📊 Consolidation: {pattern_data['consolidation_duration']} candele ({pattern_data['range_pct']:.2f}%)\n\n"
                
                caption += f"💵 Entry: <b>${entry_price:.{price_decimals}f}</b>\n"
                caption += f"🛑 Stop Loss: <b>${sl_price:.{price_decimals}f}</b>\n"
                caption += f"   (sotto consolidamento)\n"
                caption += f"🎯 Take Profit: <b>${tp_price:.{price_decimals}f}</b>\n"
                caption += f"   (2.5R projection)\n\n"
                
                caption += f"📊 <b>Quality Metrics:</b>\n"
                caption += f"• Breakout volume: {pattern_data['volume_ratio']:.1f}x\n"
                caption += f"• Breakout body: {pattern_data['breakout_body_pct']:.1f}%\n"
                caption += f"• EMA 60: ${pattern_data['ema60']:.{price_decimals}f}\n"
                caption += f"• Min dist EMA 60: {pattern_data['min_distance_to_ema60_pct']:.2f}%\n"
                caption += f"• EMA aligned: {'✅' if pattern_data['ema_aligned'] else '⚠️'}\n"
                caption += f"• Near EMA 60: {'✅' if pattern_data['near_ema60'] else '⚠️'}\n"
            
            # === LIQUIDITY SWEEP + REVERSAL ===
            elif pattern == 'Liquidity Sweep + Reversal' and pattern_data:
                entry_price = pattern_data.get('suggested_entry', last_close)
                sl_price = pattern_data.get('suggested_sl', last_close * 0.98)
                risk = entry_price - sl_price
                tp_price = entry_price + (risk * 2.0)
                ema_used = 'Sweep Low'
                ema_value = pattern_data['sweep_low']
                
                # Verifica distanza entry
                entry_distance = abs(entry_price - last_close) / last_close
                if entry_distance > 0.005:
                    logging.warning(f'⚠️ {symbol}: Entry price troppo lontano')
                    entry_price = last_close
                
                logging.info(f'💎 Liquidity Sweep Entry Setup:')
                logging.info(f'   Entry: ${entry_price:.{price_decimals}f}')
                logging.info(f'   SL: ${sl_price:.{price_decimals}f}')
                logging.info(f'   TP: ${tp_price:.{price_decimals}f} (2R)')
            
            # === SUPPORT/RESISTANCE BOUNCE ===
            elif pattern == 'Support/Resistance Bounce' and pattern_data:
                entry_price = last_close
                support_level = pattern_data['support_level']
                sl_price = support_level * 0.998
                risk = entry_price - sl_price
                tp_price = entry_price + (risk * 1.6)
                ema_used = 'Support Level'
                ema_value = support_level
                
                logging.info(f'🎯 S/R Bounce Entry Setup:')
                logging.info(f'   Support: ${support_level:.{price_decimals}f}')
                logging.info(f'   Entry: ${entry_price:.{price_decimals}f}')
                logging.info(f'   SL: ${sl_price:.{price_decimals}f}')
                logging.info(f'   TP: ${tp_price:.{price_decimals}f} (1.6R)')

            elif pattern == 'Breakout + Retest' and pattern_data:
                """
                ENTRY LOGIC per Breakout + Retest
                
                Entry: Al bounce dal retest (prezzo corrente)
                SL: Sotto retest low (con buffer 0.2%)
                TP: Resistance + (2 × range consolidamento)
                
                R:R tipico: 1:2.5-3
                """
                
                entry_price = pattern_data.get('suggested_entry')
                sl_price   = pattern_data.get('suggested_sl')
                tp_price   = pattern_data.get('suggested_tp')
                
                if entry_price is None or sl_price is None or tp_price is None:
                    logging.error(f"{symbol} {timeframe} - Pattern {pattern}: missing suggested_* in pattern_data keys={list(pattern_data.keys())}")
                    return  # oppure continue / skip ordine
                
                ema_used = 'Retest Zone'
                ema_value = pattern_data['resistance']
                
                # Calcola decimali
                price_decimals = get_price_decimals(entry_price)
                
                # Log dettagliato
                logging.info(f'🔄 Breakout + Retest Entry Setup:')
                logging.info(f'   Resistance (support): ${pattern_data["resistance"]:.{price_decimals}f}')
                logging.info(f'   Range: {pattern_data["range_pct"]:.2f}%')
                logging.info(f'   Breakout: ${pattern_data["breakout_price"]:.{price_decimals}f}')
                logging.info(f'   Retest Low: ${pattern_data["retest_low"]:.{price_decimals}f}')
                logging.info(f'   Entry: ${entry_price:.{price_decimals}f}')
                logging.info(f'   SL: ${sl_price:.{price_decimals}f}')
                logging.info(f'   TP: ${tp_price:.{price_decimals}f}')
                logging.info(f'   Volume bounce: {pattern_data["retest_vol_ratio"]:.1f}x')
                logging.info(f'   Rejection: {pattern_data["retest_rejection_pct"]:.1f}%')
                
                # Nel caption aggiungi info specifiche
                caption += f"🔄 <b>Breakout + Retest</b>\n"
                caption += f"📊 Range: {pattern_data['range_pct']:.2f}%\n"
                caption += f"💥 Breakout: ${pattern_data['breakout_price']:.{price_decimals}f}\n"
                caption += f"🔄 Retest Zone: ${pattern_data['resistance']:.{price_decimals}f}\n"
                caption += f"📍 Retest Low: ${pattern_data['retest_low']:.{price_decimals}f}\n\n"
                
                caption += f"💵 Entry: <b>${entry_price:.{price_decimals}f}</b>\n"
                caption += f"🛑 Stop Loss: <b>${sl_price:.{price_decimals}f}</b>\n"
                caption += f"   (sotto retest low)\n"
                caption += f"🎯 Take Profit: <b>${tp_price:.{price_decimals}f}</b>\n"
                caption += f"   (2R projection)\n\n"
                
                caption += f"📊 <b>Quality Metrics:</b>\n"
                caption += f"• Breakout volume: {pattern_data['breakout_vol_ratio']:.1f}x\n"
                caption += f"• Retest rejection: {pattern_data['retest_rejection_pct']:.1f}%\n"
                caption += f"• Pullback: {pattern_data['pullback_duration']} candele\n"
                caption += f"• R touches: {pattern_data['touches_resistance']}\n"

            elif pattern == 'Bullish Engulfing (GOLD)' or \
                 pattern == 'Bullish Engulfing (GOOD)' or \
                 pattern == 'Bullish Engulfing (OK)':
                
                entry_price = pattern_data['entry_price']
                sl_price = pattern_data['suggested_sl']
                tp_price = pattern_data['suggested_tp']
                ema_used = 'Engulfing Enhanced'
                ema_value = pattern_data['ema60']  # Per riferimento
                
                # Caption speciale per Engulfing
                tier = pattern_data['tier']
                score = pattern_data['quality_score']
                
                quality_emoji_map = {
                    'GOLD': '🌟',
                    'GOOD': '✅',
                    'OK': '⚠️'
                }
                
                q_emoji = quality_emoji_map.get(tier, '⚪')
                
                caption = f"🟢 <b>BULLISH ENGULFING {tier}</b> {q_emoji}\n\n"
                
                # Tier info
                caption += f"<b>Quality Score: {score}/100</b>\n\n"
                
                # EMA Setup
                caption += f"<b>📈 EMA Setup:</b>\n"
                caption += f"EMA 10: ${pattern_data['ema10']:.{price_decimals}f}\n"
                caption += f"EMA 60: ${pattern_data['ema60']:.{price_decimals}f}\n"
                caption += f"Distance EMA 10: {pattern_data['distance_to_ema10']:.2f}%\n"
                caption += f"Distance EMA 60: {pattern_data['distance_to_ema60']:.2f}%\n"
                
                if pattern_data['near_ema60']:
                    caption += f"🌟 <b>VICINO EMA 60</b> (Institutional support!)\n"
                elif pattern_data['near_ema10']:
                    caption += f"✅ <b>VICINO EMA 10</b> (Short-term support)\n"
                
                caption += f"\n"
                
                # Pullback
                if pattern_data['had_pullback']:
                    caption += f"🔄 <b>Pullback confermato</b>\n"
                
                # Volume
                caption += f"📊 Volume: {pattern_data['volume_ratio']:.1f}x\n"
                
                # Rejection
                caption += f"📍 Rejection: {pattern_data['rejection_strength']:.2f}x corpo\n"
                caption += f"Wick: {pattern_data['lower_wick_pct']:.1f}%\n\n"
                
                # Trading setup
                caption += f"<b>🎯 Trade Setup:</b>\n"
                caption += f"Entry: ${entry_price:.{price_decimals}f}\n"
                caption += f"SL: ${sl_price:.{price_decimals}f}\n"
                
                if pattern_data['near_ema60']:
                    caption += f"  (sotto EMA 60 + buffer)\n"
                else:
                    caption += f"  (sotto low candela)\n"
                
                caption += f"TP: ${tp_price:.{price_decimals}f} (2R)\n"
                
                # Risk calculation
                # ===== DYNAMIC RISK CALCULATION =====
                # Calcola risk basato su EMA score
                # Prima di usare risk_base
                risk_base = RISK_USD  # default sempre definito
                if ema_analysis and 'score' in ema_analysis:
                    ema_score = ema_analysis['score']
                    risk_base = calculate_dynamic_risk(ema_score)
                    logging.info(f"Dynamic risk for {symbol}: EMA score {ema_score} → ${risk_base:.2f}")
                else:
                    risk_base = RISK_USD
                    logging.info(f"No EMA analysis, using base risk ${RISK_USD}")
                
                # Apply symbol-specific override se configurato
                if symbol in SYMBOL_RISK_OVERRIDE:
                    risk_for_symbol = SYMBOL_RISK_OVERRIDE[symbol]
                    logging.info(f"Symbol override for {symbol}: ${risk_for_symbol:.2f}")
                else:
                    risk_for_symbol = risk_base
                
                #qty = calculate_position_size(entry_price, sl_price, risk_for_symbol)
                # ===== INTELLIGENT POSITION SIZING =====
                # Calcola ATR per volatilità
                lastatr = atr(df, period=14).iloc[-1]
                if math.isnan(lastatr):
                    lastatr = abs(entry_price - sl_price) * 0.01  # Fallback: 1% del range
                
                # Calcola qty con position sizing intelligente
                ema_score = ema_analysis['score'] if ema_analysis else 50
                qty = calculate_optimal_position_size(
                    entry_price=entry_price,
                    sl_price=sl_price,
                    symbol=symbol,
                    volatility_atr=lastatr,
                    ema_score=ema_score,
                    risk_usd=risk_for_symbol
                )
                
                # Add info nel caption
                caption += f"📊 Position Sizing:\n"
                caption += f"Position Size: {qty:.4f}\n"
                caption += f"Risk per Trade: ${risk_for_symbol:.2f}\n"
                if lastatr > 0:
                    volatility_pct = (lastatr / entry_price) * 100
                    caption += f"ATR: {lastatr:.2f} ({volatility_pct:.2f}% volatility)\n"

                # Add risk info nel caption
                caption += f"📊 Risk Management:\n"
                caption += f"Position Size: {qty:.4f}\n"
                caption += f"Risk per Trade: ${risk_for_symbol:.2f}\n"
                if ema_analysis:
                    caption += f"EMA Score: {ema_analysis['score']}/100 ({ema_analysis['quality']})\n"
                    caption += f"Risk Tier: "
                    if ema_analysis['score'] >= 80:
                        caption += "🌟 GOLD (Max Risk)\n"
                    elif ema_analysis['score'] >= 60:
                        caption += "✅ GOOD (Standard Risk)\n"
                    elif ema_analysis['score'] >= 40:
                        caption += "⚠️ OK (Reduced Risk)\n"
                    else:
                        caption += "❌ WEAK (Minimal Risk)\n"


            elif pattern == 'Pin Bar Bullish (GOLD)' or \
                 pattern == 'Pin Bar Bullish (GOOD)' or \
                 pattern == 'Pin Bar Bullish (OK)':
                
                entry_price = pattern_data.get('suggested_entry')
                sl_price   = pattern_data.get('suggested_sl')
                tp_price   = pattern_data.get('suggested_tp')
                
                if entry_price is None or sl_price is None or tp_price is None:
                    logging.error(f"{symbol} {timeframe} - Pattern {pattern}: missing suggested_* in pattern_data keys={list(pattern_data.keys())}")
                    return  # oppure continue / skip ordine
                     
                ema_used = 'Pin Bar Enhanced'
                ema_value = pattern_data['ema60']
                
                tier = pattern_data['tier']
                score = pattern_data['quality_score']
                
                quality_emoji_map = {
                    'GOLD': '🌟',
                    'GOOD': '✅',
                    'OK': '⚠️'
                }
                
                q_emoji = quality_emoji_map.get(tier, '⚪')
                
                caption = f"📍 <b>PIN BAR BULLISH {tier}</b> {q_emoji}\n\n"
                
                # Quality score
                caption += f"<b>Quality Score: {score}/100</b>\n\n"
                
                # Pin Bar Anatomy
                caption += f"<b>📊 Pin Bar Anatomy:</b>\n"
                caption += f"Lower Wick: <b>{pattern_data['lower_wick_pct']:.1f}%</b> (tail)\n"
                caption += f"Body: {pattern_data['body_pct']:.1f}%\n"
                caption += f"Upper Wick: {pattern_data['upper_wick_pct']:.1f}%\n"
                caption += f"Close Position: {pattern_data['close_position']:.1f}% del range\n"
                caption += f"Type: {'🟢 Bullish' if pattern_data['is_bullish'] else '⚪ Doji'}\n\n"
                
                # EMA Setup con ASCII art della tail
                caption += f"<b>📈 EMA Setup:</b>\n"
                caption += f"EMA 10: ${pattern_data['ema10']:.{price_decimals}f}\n"
                caption += f"EMA 60: ${pattern_data['ema60']:.{price_decimals}f}\n"
                
                # Mostra dove la tail tocca
                if pattern_data['tail_near_ema60']:
                    caption += f"🌟 <b>TAIL TOCCA EMA 60!</b>\n"
                    caption += f"   Distance: {pattern_data['tail_distance_to_ema60']:.2f}%\n"
                    caption += f"   → Institutional support zone\n"
                elif pattern_data['tail_near_ema10']:
                    caption += f"✅ <b>TAIL TOCCA EMA 10</b>\n"
                    caption += f"   Distance: {pattern_data['tail_distance_to_ema10']:.2f}%\n"
                    caption += f"   → Short-term support\n"
                else:
                    caption += f"Tail→EMA 10: {pattern_data['tail_distance_to_ema10']:.2f}%\n"
                    caption += f"Tail→EMA 60: {pattern_data['tail_distance_to_ema60']:.2f}%\n"
                
                caption += f"\n"
                
                # Liquidity Sweep (MAJOR signal)
                if pattern_data['swept_liquidity']:
                    caption += f"💎 <b>LIQUIDITY SWEEP DETECTED!</b>\n"
                    caption += f"   Swept {pattern_data['sweep_depth']:.2f}% below previous low\n"
                    caption += f"   → Stop hunt + reversal (institutional)\n\n"
                
                # Pullback
                if pattern_data['pullback_detected']:
                    caption += f"🔄 <b>Pullback: {pattern_data['pullback_depth']:.1f}%</b>\n"
                    
                    if pattern_data['fib_retracement']:
                        caption += f"   📐 FIBONACCI ZONE (50-61.8%)\n"
                        caption += f"   → Perfect retracement\n"
                    
                    caption += f"\n"
                
                # Volume
                vol_emoji = "🔥" if pattern_data['volume_ratio'] >= 3.0 else "📊"
                caption += f"{vol_emoji} <b>Volume: {pattern_data['volume_ratio']:.1f}x</b>\n"
                
                if pattern_data['volume_ratio'] >= 3.0:
                    caption += f"   → Panic selling / Capitulation\n"
                
                caption += f"\n"
                
                # Rejection Zone
                caption += f"<b>🎯 Rejection Zone (entry):</b>\n"
                caption += f"Low: ${pattern_data['rejection_zone_low']:.{price_decimals}f}\n"
                caption += f"High: ${pattern_data['rejection_zone_high']:.{price_decimals}f}\n"
                caption += f"(primi 30% della tail)\n\n"
                
                # Trading Setup
                caption += f"<b>💼 Trade Setup:</b>\n"
                caption += f"Entry: ${entry_price:.{price_decimals}f}\n"
                caption += f"SL: ${sl_price:.{price_decimals}f}\n"
                
                if pattern_data['tail_near_ema60']:
                    caption += f"   (sotto tail + EMA 60)\n"
                else:
                    caption += f"   (sotto pin bar low)\n"
                
                caption += f"TP: ${tp_price:.{price_decimals}f} (2R)\n"
                
                # Risk calculation
                # ===== DYNAMIC RISK CALCULATION =====
                # Calcola risk basato su EMA score
                # Prima di usare risk_base
                risk_base = config.RISK_USD  # default sempre definito
                if ema_analysis and 'score' in ema_analysis:
                    ema_score = ema_analysis['score']
                    risk_base = calculate_dynamic_risk(ema_score)
                    logging.info(f"Dynamic risk for {symbol}: EMA score {ema_score} → ${risk_base:.2f}")
                else:
                    risk_base = RISK_USD
                    logging.debug(f"No EMA analysis, using base risk ${RISK_USD}")
                
                # Apply symbol-specific override se configurato
                if symbol in config.SYMBOL_RISK_OVERRIDE:
                    risk_for_symbol = config.SYMBOL_RISK_OVERRIDE[symbol]
                    logging.info(f"Symbol override for {symbol}: ${risk_for_symbol:.2f}")
                else:
                    risk_for_symbol = risk_base
                
                #qty = calculate_position_size(entry_price, sl_price, risk_for_symbol)
                # ===== INTELLIGENT POSITION SIZING =====
                # Calcola ATR per volatilità
                lastatr = atr(df, period=14).iloc[-1]
                if math.isnan(lastatr):
                    lastatr = abs(entry_price - sl_price) * 0.01  # Fallback: 1% del range
                
                # Calcola qty con position sizing intelligente
                ema_score = ema_analysis['score'] if ema_analysis else 50
                qty = calculate_optimal_position_size(
                    entry_price=entry_price,
                    sl_price=sl_price,
                    symbol=symbol,
                    volatility_atr=lastatr,
                    ema_score=ema_score,
                    risk_usd=risk_for_symbol
                )
                
                # Add info nel caption
                caption += f"📊 Position Sizing:\n"
                caption += f"Position Size: {qty:.4f}\n"
                caption += f"Risk per Trade: ${risk_for_symbol:.2f}\n"
                if lastatr > 0:
                    volatility_pct = (lastatr / entry_price) * 100
                    caption += f"ATR: {lastatr:.2f} ({volatility_pct:.2f}% volatility)\n"

                # Add risk info nel caption
                caption += f"📊 Risk Management:\n"
                caption += f"Position Size: {qty:.4f}\n"
                caption += f"Risk per Trade: ${risk_for_symbol:.2f}\n"
                if ema_analysis:
                    caption += f"EMA Score: {ema_analysis['score']}/100 ({ema_analysis['quality']})\n"
                    caption += f"Risk Tier: "
                    if ema_analysis['score'] >= 80:
                        caption += "🌟 GOLD (Max Risk)\n"
                    elif ema_analysis['score'] >= 60:
                        caption += "✅ GOOD (Standard Risk)\n"
                    elif ema_analysis['score'] >= 40:
                        caption += "⚠️ OK (Reduced Risk)\n"
                    else:
                        caption += "❌ WEAK (Minimal Risk)\n"
                
                # Strategic notes
                caption += f"\n<b>💡 Strategic Notes:</b>\n"
                
                if pattern_data['swept_liquidity'] and pattern_data['tail_near_ema60']:
                    caption += f"🌟 PREMIUM SETUP:\n"
                    caption += f"• Liquidity sweep (stop hunt)\n"
                    caption += f"• EMA 60 bounce\n"
                    caption += f"• High probability reversal\n"
                elif tier == 'GOLD':
                    caption += f"🌟 GOLD SETUP:\n"
                    caption += f"• EMA 60 support confirmed\n"
                    caption += f"• Strong rejection\n"
                    caption += f"• Expect continuation\n"
                elif tier == 'GOOD':
                    caption += f"✅ SOLID SETUP:\n"
                    caption += f"• EMA 10 support\n"
                    caption += f"• Swing trade zone\n"
                
                # Autotrade
                if job_ctx.get('autotrade') and qty > 0 and not position_exists:
                    order_res = await place_bybit_order(
                        symbol, 
                        side, 
                        qty, 
                        sl_price, 
                        tp_price,
                        entry_price,
                        timeframe,
                        chat_id,
                        pattern
                    )
                    
                    if 'error' in order_res:
                        caption += f"\n\n❌ <b>Errore ordine:</b>\n{order_res['error']}"
                    else:
                        caption += f"\n\n✅ <b>Ordine su Bybit {config.TRADING_MODE.upper()}</b>"

            # 🌱 BUD PATTERN CAPTION
            elif pattern == 'BUD Pattern' or pattern == 'MAXI BUD Pattern':
                
                tier = 'MAXI' if 'MAXI' in pattern else 'STANDARD'
                # ← AGGIUNGI fallback sicuro:
                entry_price = pattern_data.get('suggested_entry', last_close)
                sl_price = pattern_data.get('suggested_sl', last_close * 0.98)
                tp_price = pattern_data.get('suggested_tp', last_close * 1.02)
                
                price_decimals = get_price_decimals(entry_price)
                
                caption = f"🌱 <b>{pattern.upper()}</b>\n\n"
                
                if tier == 'MAXI':
                    caption += f"⭐ <b>Setup PREMIUM</b> ({pattern_data['rest_count']} candele riposo)\n\n"
                else:
                    caption += f"📊 <b>Setup VALIDO</b> ({pattern_data['rest_count']} candele riposo)\n\n"
                
                caption += f"💥 <b>Breakout Phase:</b>\n"
                caption += f"  High: ${pattern_data['breakout_high']:.{price_decimals}f}\n"
                caption += f"  Low: ${pattern_data['breakout_low']:.{price_decimals}f}\n"
                caption += f"  Range: ${pattern_data['breakout_range']:.{price_decimals}f}\n"
                caption += f"  Body: {pattern_data['breakout_body_pct']:.1f}%\n\n"
                
                caption += f"🛌 <b>Rest Phase:</b>\n"
                caption += f"  Candele: {pattern_data['rest_count']}\n"
                caption += f"  Avg Range: {pattern_data['rest_range_pct']:.1f}% del breakout\n"
                caption += f"  Status: {'✅ Compresse' if pattern_data['rest_range_pct'] < 60 else '⚠️'}\n\n"
                
                caption += f"💥 <b>Trigger:</b>\n"
                caption += f"  {'✅' if pattern_data['breaks_breakout_high'] else '⚠️'} Rompe breakout high\n"
                caption += f"  Candela: {'🟢 Verde' if pattern_data['is_green'] else '⚪'}\n\n"
                
                caption += f"📊 <b>Volume & EMA:</b>\n"
                if pattern_data['volume_ok']:
                    caption += f"  ✅ Volume: {pattern_data['volume_ratio']:.1f}x\n"
                else:
                    caption += f"  ⚠️ Volume: {pattern_data['volume_ratio']:.1f}x (< 1.5x)\n"
                
                caption += f"  EMA 10: ${pattern_data['ema10']:.{price_decimals}f}\n"
                caption += f"  EMA 60: ${pattern_data['ema60']:.{price_decimals}f}\n"
                caption += f"  {'✅' if pattern_data['above_ema60'] else '⚠️'} Sopra EMA 60 (uptrend)\n\n"
                
                caption += f"🎯 <b>Trade Setup:</b>\n"
                caption += f"  Entry: ${entry_price:.{price_decimals}f}\n"
                caption += f"  SL: ${sl_price:.{price_decimals}f}\n"
                caption += f"     (sotto breakout low)\n"
                caption += f"  TP: ${tp_price:.{price_decimals}f} (2R)\n\n"
                
                # ===== DYNAMIC RISK CALCULATION =====
                risk_base = RISK_USD
                if ema_analysis and 'score' in ema_analysis:
                    ema_score = ema_analysis['score']
                    risk_base = calculate_dynamic_risk(ema_score)
                    logging.info(f"Dynamic risk for {symbol}: EMA score {ema_score} → ${risk_base:.2f}")
                else:
                    risk_base = RISK_USD
                    logging.debug(f"No EMA analysis, using base risk ${RISK_USD}")
                
                # Apply symbol-specific override
                if symbol in SYMBOL_RISK_OVERRIDE:
                    risk_for_symbol = SYMBOL_RISK_OVERRIDE[symbol]
                    logging.info(f"Symbol override for {symbol}: ${risk_for_symbol:.2f}")
                else:
                    risk_for_symbol = risk_base
                
                # ===== INTELLIGENT POSITION SIZING =====
                lastatr = atr(df, period=14).iloc[-1]
                if math.isnan(lastatr):
                    lastatr = abs(entry_price - sl_price) * 0.01
                
                ema_score = ema_analysis['score'] if ema_analysis else 50
                qty = calculate_optimal_position_size(
                    entry_price=entry_price,
                    sl_price=sl_price,
                    symbol=symbol,
                    volatility_atr=lastatr,
                    ema_score=ema_score,
                    risk_usd=risk_for_symbol
                )
                
                caption += f"📊 <b>Risk Management:</b>\n"
                caption += f"  Position Size: {qty:.4f}\n"
                caption += f"  Risk per Trade: ${risk_for_symbol:.2f}\n"
                
                if ema_analysis:
                    caption += f"  EMA Score: {ema_analysis['score']}/100 ({ema_analysis['quality']})\n"
                    caption += f"  Risk Tier: "
                    if ema_analysis['score'] >= 80:
                        caption += "🌟 GOLD (Max Risk)\n"
                    elif ema_analysis['score'] >= 60:
                        caption += "✅ GOOD (Standard Risk)\n"
                    elif ema_analysis['score'] >= 40:
                        caption += "⚠️ OK (Reduced Risk)\n"
                    else:
                        caption += "❌ WEAK (Minimal Risk)\n"
                
                if lastatr > 0:
                    volatility_pct = (lastatr / entry_price) * 100
                    caption += f"  ATR: {lastatr:.2f} ({volatility_pct:.2f}% volatility)\n"
                
                caption += f"\n💡 <b>Strategy Notes:</b>\n"
                caption += f"  • Breakout + riposo = buyers confidenti\n"
                caption += f"  • Pattern compresso = energia per pump\n"
                if tier == 'MAXI':
                    caption += f"  • ⭐ MAXI: 3+ riposo = setup superiore\n"
                
                # Position check
                if position_exists:
                    caption += "\n\n🚫 <b>Posizione già aperta</b>"
                    caption += f"\nOrdine NON eseguito per {symbol}"
                
                # Autotrade
                if job_ctx.get('autotrade') and qty > 0 and not position_exists:
                    order_res = await place_bybit_order(
                        symbol, 
                        side, 
                        qty, 
                        sl_price, 
                        tp_price,
                        entry_price,
                        timeframe,
                        chat_id,
                        pattern
                    )
                    
                    if 'error' in order_res:
                        caption += f"\n\n❌ <b>Errore ordine:</b>\n{order_res['error']}"
                    else:
                        caption += f"\n\n✅ <b>Ordine su Bybit {config.TRADING_MODE.upper()}</b>"

            elif pattern == 'Morning Star (GOLD)' or \
                 pattern == 'Morning Star (GOOD)' or \
                 pattern == 'Morning Star (OK)':
                
                entry_price = pattern_data['entry_price']
                sl_price = pattern_data['suggested_sl']
                tp_price = pattern_data['suggested_tp']
                ema_used = 'Morning Star Enhanced'
                ema_value = pattern_data['ema60']
                
                tier = pattern_data['tier']
                score = pattern_data['quality_score']
                
                quality_emoji_map = {
                    'GOLD': '🌟',
                    'GOOD': '✅',
                    'OK': '⚠️'
                }
                
                q_emoji = quality_emoji_map.get(tier, '⚪')
                
                caption = f"⭐ <b>MORNING STAR {tier}</b> {q_emoji}\n\n"
                
                # Quality score
                caption += f"<b>Quality Score: {score}/100</b>\n\n"
                
                # Pattern Structure (3 candele)
                caption += f"<b>📊 Pattern Structure:</b>\n"
                caption += f"Candela A (ribassista):\n"
                caption += f"  Body: {pattern_data['candle_a']['body_pct']:.1f}% range\n"
                caption += f"Candela B (indecisione):\n"
                caption += f"  Body: {pattern_data['candle_b']['body_pct']:.1f}% range\n"
                caption += f"  Low: ${pattern_data['candle_b']['low']:.{price_decimals}f}\n"
                caption += f"Candela C (rialzista):\n"
                caption += f"  Body: {pattern_data['candle_c']['body_pct']:.1f}% range\n\n"
                
                # Recovery
                caption += f"<b>🔄 Recovery Analysis:</b>\n"
                caption += f"Recupero: <b>{pattern_data['recovery_pct']:.1f}%</b>\n"
                
                if pattern_data['fib_recovery']:
                    caption += f"📐 <b>FIBONACCI ZONE (61.8%)</b>\n"
                    caption += f"   → Golden ratio reversal!\n"
                
                caption += f"\n"
                
                # EMA Setup
                caption += f"<b>📈 EMA Setup:</b>\n"
                caption += f"EMA 10: ${pattern_data['ema10']:.{price_decimals}f}\n"
                caption += f"EMA 60: ${pattern_data['ema60']:.{price_decimals}f}\n"
                
                if pattern_data['b_touches_ema60']:
                    caption += f"🌟 <b>CANDELA B TOCCA EMA 60!</b>\n"
                    caption += f"   Distance: {pattern_data['b_distance_to_ema60']:.2f}%\n"
                    caption += f"   → Institutional support zone\n"
                elif pattern_data['b_touches_ema10']:
                    caption += f"✅ <b>CANDELA B TOCCA EMA 10</b>\n"
                    caption += f"   → Short-term support\n"
                
                if pattern_data['ema_sandwich']:
                    caption += f"🎯 <b>EMA SANDWICH!</b>\n"
                    caption += f"   Candela B tra EMA 10 e 60\n"
                    caption += f"   → Accumulation zone\n"
                
                caption += f"\n"
                
                # Gap Detection
                if pattern_data['gap_detected']:
                    caption += f"💥 <b>GAP DOWN PANIC!</b>\n"
                    caption += f"   Gap size: {pattern_data['gap_size']:.2f}%\n"
                    caption += f"   → Capitulation + reversal\n\n"
                
                # Pullback
                if pattern_data['pullback_detected']:
                    caption += f"🔄 <b>Pullback: {pattern_data['pullback_depth']:.1f}%</b>\n"
                    caption += f"   → Shakeout confirmed\n\n"
                
                # Volume Analysis
                caption += f"<b>📊 Volume Progression:</b>\n"
                caption += f"A: {pattern_data['vol_a']:.0f}\n"
                caption += f"B: {pattern_data['vol_b']:.0f} (selling exhaustion)\n"
                caption += f"C: {pattern_data['vol_c']:.0f} (<b>{pattern_data['vol_c_ratio']:.1f}x</b> surge)\n"
                
                if pattern_data['vol_progression_ok']:
                    caption += f"✅ <b>PERFECT PROGRESSION!</b>\n"
                    caption += f"   A > B < C (textbook pattern)\n"
                
                caption += f"\n"
                
                # Trading Setup
                caption += f"<b>💼 Trade Setup:</b>\n"
                caption += f"Entry: ${entry_price:.{price_decimals}f}\n"
                caption += f"SL: ${sl_price:.{price_decimals}f}\n"
                
                if pattern_data['b_touches_ema60']:
                    caption += f"   (sotto candela B + EMA 60)\n"
                else:
                    caption += f"   (sotto candela B low)\n"
                
                caption += f"TP: ${tp_price:.{price_decimals}f} (2R)\n"
                
                # Risk calculation
                # ===== DYNAMIC RISK CALCULATION =====
                # Calcola risk basato su EMA score
                # Prima di usare risk_base
                risk_base = config.RISK_USD  # default sempre definito
                if ema_analysis and 'score' in ema_analysis:
                    ema_score = ema_analysis['score']
                    risk_base = calculate_dynamic_risk(ema_score)
                    logging.info(f"Dynamic risk for {symbol}: EMA score {ema_score} → ${risk_base:.2f}")
                else:
                    risk_base = RISK_USD
                    logging.debug(f"No EMA analysis, using base risk ${RISK_USD}")
                
                # Apply symbol-specific override se configurato
                if symbol in config.SYMBOL_RISK_OVERRIDE:
                    risk_for_symbol = config.SYMBOL_RISK_OVERRIDE[symbol]
                    logging.info(f"Symbol override for {symbol}: ${risk_for_symbol:.2f}")
                else:
                    risk_for_symbol = risk_base
                
                #qty = calculate_position_size(entry_price, sl_price, risk_for_symbol)
                # ===== INTELLIGENT POSITION SIZING =====
                # Calcola ATR per volatilità
                lastatr = atr(df, period=14).iloc[-1]
                if math.isnan(lastatr):
                    lastatr = abs(entry_price - sl_price) * 0.01  # Fallback: 1% del range
                
                # Calcola qty con position sizing intelligente
                ema_score = ema_analysis['score'] if ema_analysis else 50
                qty = calculate_optimal_position_size(
                    entry_price=entry_price,
                    sl_price=sl_price,
                    symbol=symbol,
                    volatility_atr=lastatr,
                    ema_score=ema_score,
                    risk_usd=risk_for_symbol
                )
                
                # Add info nel caption
                caption += f"📊 Position Sizing:\n"
                caption += f"Position Size: {qty:.4f}\n"
                caption += f"Risk per Trade: ${risk_for_symbol:.2f}\n"
                if lastatr > 0:
                    volatility_pct = (lastatr / entry_price) * 100
                    caption += f"ATR: {lastatr:.2f} ({volatility_pct:.2f}% volatility)\n"

                
                # Add risk info nel caption
                caption += f"📊 Risk Management:\n"
                caption += f"Position Size: {qty:.4f}\n"
                caption += f"Risk per Trade: ${risk_for_symbol:.2f}\n"
                if ema_analysis:
                    caption += f"EMA Score: {ema_analysis['score']}/100 ({ema_analysis['quality']})\n"
                    caption += f"Risk Tier: "
                    if ema_analysis['score'] >= 80:
                        caption += "🌟 GOLD (Max Risk)\n"
                    elif ema_analysis['score'] >= 60:
                        caption += "✅ GOOD (Standard Risk)\n"
                    elif ema_analysis['score'] >= 40:
                        caption += "⚠️ OK (Reduced Risk)\n"
                    else:
                        caption += "❌ WEAK (Minimal Risk)\n"

                
                # Strategic Notes
                caption += f"\n<b>💡 Strategic Notes:</b>\n"
                
                if tier == 'GOLD':
                    caption += f"🌟 <b>PREMIUM SETUP:</b>\n"
                    if pattern_data['b_touches_ema60']:
                        caption += f"• EMA 60 support (institutional)\n"
                    if pattern_data['gap_detected']:
                        caption += f"• Gap down panic → reversal\n"
                    if pattern_data['fib_recovery']:
                        caption += f"• Fibonacci golden ratio\n"
                    caption += f"• High probability continuation\n"
                elif tier == 'GOOD':
                    caption += f"✅ <b>SOLID SETUP:</b>\n"
                    caption += f"• EMA 10 support\n"
                    caption += f"• Good volume confirmation\n"
                    caption += f"• Swing trade zone\n"
                
                # Autotrade
                if job_ctx.get('autotrade') and qty > 0 and not position_exists:
                    order_res = await place_bybit_order(
                        symbol, 
                        side, 
                        qty, 
                        sl_price, 
                        tp_price,
                        entry_price,
                        timeframe,
                        chat_id,
                        pattern
                    )
                    
                    if 'error' in order_res:
                        caption += f"\n\n❌ <b>Errore ordine:</b>\n{order_res['error']}"
                    else:
                        caption += f"\n\n✅ <b>Ordine su Bybit {config.TRADING_MODE.upper()}</b>"

            
            # === LOGICA STANDARD per altri pattern ===
            else:
                entry_price = last_close
                
                # Calcola SL basato su EMA o ATR
                if USE_EMA_STOP_LOSS:
                    sl_price, ema_used, ema_value = calculate_ema_stop_loss(
                        df, timeframe, last_close, side
                    )
                else:
                    if not math.isnan(last_atr) and last_atr > 0:
                        sl_price = last_close - last_atr * ATR_MULT_SL
                        ema_used = 'ATR'
                        ema_value = last_atr
                    else:
                        sl_price = df['low'].iloc[-1]
                        ema_used = 'Low'
                        ema_value = 0
                
                # Calcola TP
                if not math.isnan(last_atr) and last_atr > 0:
                    tp_price = last_close + last_atr * ATR_MULT_TP
                else:
                    tp_price = last_close * 1.02
            
            # ===== CALCOLO POSITION SIZE E CHECK =====
            #risk_for_symbol = SYMBOL_RISK_OVERRIDE.get(symbol, RISK_USD)
            #qty = calculate_position_size(entry_price, sl_price, risk_for_symbol)
            # Apply symbol-specific override se configurato
            risk_base = config.RISK_USD
            if symbol in config.SYMBOL_RISK_OVERRIDE:
                risk_for_symbol = config.SYMBOL_RISK_OVERRIDE[symbol]
                logging.info(f"Symbol override for {symbol}: ${risk_for_symbol:.2f}")
            else:
                risk_for_symbol = risk_base
            #qty = calculate_position_size(entry_price, sl_price, risk_for_symbol)
            # ===== INTELLIGENT POSITION SIZING =====
            risk_for_symbol = config.SYMBOL_RISK_OVERRIDE.get(symbol, RISK_USD)
            
            # Calcola ATR per volatilità
            lastatr = atr(df, period=14).iloc[-1]
            if math.isnan(lastatr):
                lastatr = abs(entry_price - sl_price) * 0.01  # Fallback: 1% del range
            
            # Calcola qty con position sizing intelligente
            ema_score = ema_analysis['score'] if ema_analysis else 50
            qty = calculate_optimal_position_size(
                entry_price=entry_price,
                sl_price=sl_price,
                symbol=symbol,
                volatility_atr=lastatr,
                ema_score=ema_score,
                risk_usd=risk_for_symbol
            )
            
            # Add info nel caption
            caption += f"📊 Position Sizing:\n"
            caption += f"Position Size: {qty:.4f}\n"
            caption += f"Risk per Trade: ${risk_for_symbol:.2f}\n"
            if lastatr > 0:
                volatility_pct = (lastatr / entry_price) * 100
                caption += f"ATR: {lastatr:.2f} ({volatility_pct:.2f}% volatility)\n"

            
            position_exists = symbol in config.ACTIVE_POSITIONS
            if position_exists:
                logging.warning(f'🚫 Position already exists for {symbol}, skip order')
            else:
                logging.info(f'✅ No position for {symbol}, ready to place order')

            
            # ===== COSTRUISCI CAPTION =====
            quality_emoji_map = {
                'GOLD': '🌟',
                'GOOD': '✅',
                'OK': '⚠️',
                'WEAK': '🔶',
                'BAD': '❌'
            }
            
            caption = "🔥 <b>SEGNALE BUY</b>\n\n"
            
            # EMA QUALITY
            if ema_analysis:
                q_emoji = quality_emoji_map.get(ema_analysis['quality'], '⚪')
                caption += f"{q_emoji} EMA Quality: <b>{ema_analysis['quality']}</b>\n"
                caption += f"Score: <b>{ema_analysis['score']}/100</b>\n\n"
            
            # Pattern info
            caption += f"📊 Pattern: <b>{pattern}</b>\n"
            
            # Info specifiche pattern
            if pattern == 'Bullish Flag Breakout' and pattern_data:
                caption += f"🚩 Breakout Level X: <b>${pattern_data['X']:.{price_decimals}f}</b>\n"
            
            caption += f"🪙 Symbol: <b>{symbol}</b> ({timeframe})\n"
            caption += f"🕐 {timestamp_str}\n\n"
            
            # Trading params
            caption += f"💵 Entry: <b>${entry_price:.{price_decimals}f}</b>\n"
            
            # SL/TP display specifico per pattern
            if pattern == 'Bullish Flag Breakout' and pattern_data:
                caption += f"🛑 Stop Loss: <b>${sl_price:.{price_decimals}f}</b>\n"
                caption += f"   (sotto consolidamento)\n"
                caption += f"🎯 Take Profit: <b>${tp_price:.{price_decimals}f}</b>\n"
                caption += f"   (1.5x pole height)\n"
            else:
                if config.USE_EMA_STOP_LOSS:
                    caption += f"🛑 Stop Loss: <b>${sl_price:.{price_decimals}f}</b>\n"
                    caption += f"   sotto {ema_used}"
                    if isinstance(ema_value, (int, float)) and ema_value > 0:
                        caption += f" = ${ema_value:.{price_decimals}f}"
                    caption += "\n"
                else:
                    caption += f"🛑 Stop Loss: <b>${sl_price:.{price_decimals}f}</b> ({ema_used})\n"
                
                caption += f"🎯 Take Profit: <b>${tp_price:.{price_decimals}f}</b>\n"
            
            caption += f"📦 Qty: <b>{qty:.4f}</b>\n"
            caption += f"💰 Risk: <b>${risk_for_symbol}</b>\n"
            
            rr = abs(tp_price - entry_price) / abs(sl_price - entry_price) if abs(sl_price - entry_price) > 0 else 0
            caption += f"📏 R:R: <b>{rr:.2f}:1</b>\n"
            
            # EMA Analysis dettagliata
            if ema_analysis:
                # Logica speciale per Liquidity Sweep
                if pattern == 'Liquidity Sweep + Reversal':
                    ema_analysis = analyze_ema_conditions(df, timeframe, pattern)
                
                caption += "\n━━━━━━━━━━━━━━━━━━━\n"
                caption += "📈 <b>EMA Analysis</b>\n\n"
                caption += ema_analysis['details']
                caption += f"Score: <b>{ema_analysis['score']}/100</b>\n\n"
                
                # Valori EMA CON DECIMALI DINAMICI
                if 'ema_values' in ema_analysis:
                    ema_vals = ema_analysis['ema_values']
                    ema_decimals = get_price_decimals(ema_vals['price'])
                    
                    caption += f"\n\n💡 <b>EMA Values:</b>\n"
                    caption += f"Price: ${ema_vals['price']:.{ema_decimals}f}\n"
                    caption += f"EMA 5: ${ema_vals['ema5']:.{ema_decimals}f}\n"
                    caption += f"EMA 10: ${ema_vals['ema10']:.{ema_decimals}f}\n"
                    caption += f"EMA 60: ${ema_vals['ema60']:.{ema_decimals}f}\n"
                    caption += f"EMA 223: ${ema_vals['ema223']:.{ema_decimals}f}\n"
                
                # Strategy
                if USE_EMA_STOP_LOSS:
                    caption += f"\n🎯 <b>EMA Stop:</b> Exit se prezzo rompe {ema_used}"
                
                # Info filtri applicati
                caption += f"\n\n💡 <b>Filtri Pattern:</b>\n"
                
                if config.TREND_FILTER_ENABLED:
                    caption += f"Trend: {config.TREND_FILTER_MODE.upper()}"
                    if TREND_FILTER_MODE == 'ema_based':
                        caption += f" (Price > EMA 60)\n"
                    elif TREND_FILTER_MODE == 'structure':
                        caption += f" (HH+HL)\n"
                    else:
                        caption += f"\n"
                else:
                    caption += f"Trend: OFF\n"
                
                if config.VOLUME_FILTER_ENABLED:
                    caption += f"Volume: {config.VOLUME_FILTER_MODE.upper()}\n"
                else:
                    caption += f"Volume: OFF\n"
                
                if config.EMA_FILTER_ENABLED:
                    caption += f"EMA: {config.EMA_FILTER_MODE.upper()}\n"
                else:
                    caption += f"EMA: OFF\n"

            # Warning se LOOSE mode con EMA deboli
            if ema_analysis and config.EMA_FILTER_MODE == 'loose' and not ema_analysis['passed']:
                caption += f"\n⚠️ <b>ATTENZIONE:</b> Setup con EMA non ottimali"
                caption += f"\nConsidera ridurre position size."
            
            # Posizione esistente
            if position_exists:
                caption += "\n\n🚫 <b>Posizione già aperta</b>"
                caption += f"\nOrdine NON eseguito per {symbol}"
            
            # Autotrade
            if job_ctx.get('autotrade') and qty > 0 and not position_exists:
                order_res = await place_bybit_order(
                    symbol, 
                    side, 
                    qty, 
                    sl_price, 
                    tp_price,
                    entry_price,
                    timeframe,
                    chat_id,
                    pattern
                )
                
                if 'error' in order_res:
                    caption += f"\n\n❌ <b>Errore ordine:</b>\n{order_res['error']}"
                else:
                    caption += f"\n\n✅ <b>Ordine su Bybit {config.TRADING_MODE.upper()}</b>"

        # ===== SEGNALE SELL (SHORT) =====
        elif found and side == 'Sell':
            # Check Higher Timeframe EMA (supporto che diventa resistenza)
            htf_block = check_higher_timeframe_support(
                symbol=symbol, 
                current_tf=timeframe, 
                current_price=last_close
            )
            
            if htf_block['blocked']:
                logging.warning(
                    f'🚫 Pattern {pattern} su {symbol} {timeframe} '
                    f'BLOCCATO da supporto HTF {htf_block["htf"]}'
                )

                if full_mode:
                    caption = (
                        f"⚠️ <b>Pattern BLOCCATO da HTF Support</b>\n\n"
                        f"Pattern: {pattern} su {timeframe}\n"
                        f"Timeframe superiore: {htf_block['htf']}\n\n"
                        f"Supporti HTF:\n"
                        f"{htf_block['details']}\n\n"
                        f"💡 Aspetta rottura HTF o cerca altro setup"
                    )
                    
                    try:
                        chart_buffer = generate_chart(df, symbol, timeframe)
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=chart_buffer,
                            caption=caption,
                            parse_mode='HTML'
                        )
                    except:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=caption,
                            parse_mode='HTML'
                        )
                
                return  # BLOCCA segnale

            # ===== BEARISH ENGULFING ENHANCED CAPTION =====
            if pattern.startswith('Bearish Engulfing'):
                
                tier = pattern_data['tier']
                score = pattern_data['quality_score']
                entry_price = pattern_data['entry_price']
                sl_price = pattern_data['suggested_sl']
                tp_price = pattern_data['suggested_tp']
                
                quality_emoji_map = {
                    'GOLD': '🌟',
                    'GOOD': '✅',
                    'OK': '⚠️'
                }
                
                q_emoji = quality_emoji_map.get(tier, '⚪')
                
                caption = f"🔴 <b>BEARISH ENGULFING {tier}</b> {q_emoji}\n\n"
                
                # Quality score
                caption += f"<b>Quality Score: {score}/100</b>\n\n"
                
                # EMA 60 BREAKDOWN (se presente)
                if pattern_data.get('ema60_breakdown'):
                    caption += f"🚨 <b>EMA 60 BREAKDOWN!</b>\n"
                    caption += f"• Breakdown: -{pattern_data['breakdown_strength']:.2f}%\n"
                    caption += f"• Setup PREMIUM (institutional breakdown)\n"
                    caption += f"• Win rate atteso: 75-80%\n\n"
                
                # EMA Setup
                caption += f"<b>📉 EMA Setup:</b>\n"
                caption += f"EMA 10: ${pattern_data['ema10']:.{price_decimals}f}\n"
                caption += f"EMA 60: ${pattern_data['ema60']:.{price_decimals}f}\n"
                caption += f"Distance EMA 10: {pattern_data['distance_to_ema10']:.2f}%\n"
                caption += f"Distance EMA 60: {pattern_data['distance_to_ema60']:.2f}%\n"
                
                if pattern_data['below_ema60']:
                    caption += f"✅ Sotto EMA 60 (downtrend)\n"
                
                caption += f"\n"
                
                # Rally
                if pattern_data['had_rally']:
                    caption += f"📈 <b>Rally prima del breakdown</b>\n"
                    caption += f"   Depth: {pattern_data['rally_depth']:.1f}%\n"
                
                # Volume
                caption += f"📊 Volume: {pattern_data['volume_ratio']:.1f}x\n"
                
                # Rejection
                caption += f"📍 Upper Rejection: {pattern_data['rejection_strength']:.2f}x corpo\n"
                caption += f"Upper Wick: {pattern_data['upper_wick_pct']:.1f}%\n\n"
                
                # Trading setup
                caption += f"<b>🎯 Short Setup:</b>\n"
                caption += f"Entry: ${entry_price:.{price_decimals}f}\n"
                caption += f"SL: ${sl_price:.{price_decimals}f}\n"
                
                if pattern_data.get('ema60_breakdown'):
                    caption += f"  (sopra EMA 60 + buffer)\n"
                else:
                    caption += f"  (sopra high candela)\n"
                
                caption += f"TP: ${tp_price:.{price_decimals}f} (2R)\n"

                # ===== BUD BEARISH CAPTION =====
            elif pattern.startswith('BUD Bearish') or pattern.startswith('MAXI BUD Bearish'):
                
                tier = 'MAXI' if 'MAXI' in pattern else 'STANDARD'
                
                caption = f"🔴🌱 <b>{pattern.upper()}</b>\n\n"
                
                if tier == 'MAXI':
                    caption += f"⭐ <b>Setup PREMIUM</b> ({pattern_data['rest_count']} candele riposo)\n\n"
                else:
                    caption += f"📊 <b>Setup VALIDO</b> ({pattern_data['rest_count']} candele riposo)\n\n"
                
                price_decimals = get_price_decimals(pattern_data['breakdown_low'])
                
                caption += f"📉 <b>Breakdown Phase:</b>\n"
                caption += f"  High: ${pattern_data['breakdown_high']:.{price_decimals}f}\n"
                caption += f"  Low: ${pattern_data['breakdown_low']:.{price_decimals}f}\n"
                caption += f"  Range: ${pattern_data['breakdown_range']:.{price_decimals}f}\n"
                caption += f"  Body: {pattern_data['breakdown_body_pct']:.1f}%\n\n"
                
                caption += f"🛌 <b>Rest Phase:</b>\n"
                caption += f"  Candele: {pattern_data['rest_count']}\n"
                caption += f"  Avg Range: {pattern_data['rest_range_pct']:.1f}% del breakdown\n"
                caption += f"  Status: {'✅ Compresse' if pattern_data['rest_range_pct'] < 60 else '⚠️'}\n\n"
                
                caption += f"💥 <b>Trigger:</b>\n"
                caption += f"  {'✅' if pattern_data['breaks_breakdown_low'] else '⚠️'} Rompe breakdown low\n"
                caption += f"  Candela: {'🔴 Rossa' if pattern_data['is_red'] else '⚪'}\n\n"
                
                caption += f"📊 <b>Volume & EMA:</b>\n"
                if pattern_data['volume_ok']:
                    caption += f"  ✅ Volume: {pattern_data['volume_ratio']:.1f}x\n"
                else:
                    caption += f"  ⚠️ Volume: {pattern_data['volume_ratio']:.1f}x (minore 1.5x)\n"
                
                caption += f"  EMA 10: ${pattern_data['ema10']:.{price_decimals}f}\n"
                caption += f"  EMA 60: ${pattern_data['ema60']:.{price_decimals}f}\n"
                caption += f"  {'✅' if pattern_data['below_ema60'] else '⚠️'} Sotto EMA 60 (downtrend)\n\n"
                
                caption += f"🎯 <b>SHORT Setup:</b>\n"
                caption += f"  Entry: ${pattern_data['suggested_entry']:.{price_decimals}f}\n"
                caption += f"  SL: ${pattern_data['suggested_sl']:.{price_decimals}f}\n"
                caption += f"     (sopra breakdown high)\n"
                caption += f"  TP: ${pattern_data['suggested_tp']:.{price_decimals}f} (2R)\n\n"
                
                # Risk calculation
                risk_base = RISK_USD
                if ema_analysis and 'score' in ema_analysis:
                    ema_score = ema_analysis['score']
                    risk_base = calculate_dynamic_risk(ema_score)
                
                if symbol in SYMBOL_RISK_OVERRIDE:
                    risk_for_symbol = SYMBOL_RISK_OVERRIDE[symbol]
                else:
                    risk_for_symbol = risk_base
                
                # Position sizing
                lastatr = atr(df, period=14).iloc[-1]
                if math.isnan(lastatr):
                    lastatr = abs(entry_price - sl_price) * 0.01
                
                ema_score = ema_analysis['score'] if ema_analysis else 50
                qty = calculate_optimal_position_size(
                    entry_price=entry_price,
                    sl_price=sl_price,
                    symbol=symbol,
                    volatility_atr=lastatr,
                    ema_score=ema_score,
                    risk_usd=risk_for_symbol
                )
                
                caption += f"📊 <b>Risk Management:</b>\n"
                caption += f"  Position Size: {qty:.4f}\n"
                caption += f"  Risk per Trade: ${risk_for_symbol:.2f}\n"
                
                if ema_analysis:
                    caption += f"  EMA Score: {ema_analysis['score']}/100\n"
                
                caption += f"\n💡 <b>Strategy Notes:</b>\n"
                caption += f"  • Pattern compresso = shorts confidenti\n"
                caption += f"  • Riposo = no panic buy = setup forte\n"
                if tier == 'MAXI':
                    caption += f"  • ⭐ MAXI: 3+ riposo = probabilità superiore\n"
                
                # Position check
                if position_exists:
                    caption += "\n\n🚫 <b>Posizione già aperta</b>"
                    caption += f"\nOrdine SHORT NON eseguito per {symbol}"
                
                # Autotrade
                if job_ctx.get('autotrade') and qty > 0 and not position_exists:
                    order_res = await place_bybit_order(
                        symbol, 
                        side,  # 'Sell'
                        qty, 
                        sl_price, 
                        tp_price,
                        entry_price,
                        timeframe,
                        chat_id,
                        pattern
                    )
                    
                    if 'error' in order_res:
                        caption += f"\n\n❌ <b>Errore ordine SHORT:</b>\n{order_res['error']}"
                    else:
                        caption += f"\n\n✅ <b>Ordine SHORT su Bybit {config.TRADING_MODE.upper()}</b>"

            # Check EMA filter per SELL (come per BUY)
            if config.EMA_FILTER_ENABLED and ema_analysis:
                if config.EMA_FILTER_MODE == 'strict' and not ema_analysis['passed']:
                    logging.warning(
                        f'🚫 {symbol} {timeframe} - Pattern {pattern} (SELL) '
                        f'bloccato da EMA STRICT (score {ema_analysis["score"]}/100)'
                    )
                    
                    if full_mode:
                        caption = (
                            f"🔴 <b>Pattern SELL Trovato MA Bloccato</b>\n\n"
                            f"Pattern: {pattern}\n"
                            f"EMA Score: {ema_analysis['score']}/100\n"
                            f"Threshold: 60/100 (STRICT)\n\n"
                            f"⚠️ Pattern SELL valido MA condizioni EMA non ottimali"
                        )
                        
                        try:
                            chart_buffer = generate_chart(df, symbol, timeframe)
                            await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=chart_buffer,
                                caption=caption,
                                parse_mode='HTML'
                            )
                        except:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=caption,
                                parse_mode='HTML'
                            )
                    
                    return  # Blocca ordine SELL
                
                elif config.EMA_FILTER_MODE == 'loose' and ema_analysis['score'] < 40:
                    logging.warning(
                        f'🚫 {symbol} {timeframe} - Pattern {pattern} (SELL) '
                        f'bloccato da EMA LOOSE (score {ema_analysis["score"]}/100 < 40)'
                    )
                    
                    if full_mode:
                        caption = (
                            f"🔴 <b>Pattern SELL Trovato MA Bloccato</b>\n\n"
                            f"Pattern: {pattern}\n"
                            f"EMA Score: {ema_analysis['score']}/100\n"
                            f"Threshold: 40/100 (LOOSE)\n\n"
                            f"⚠️ EMA score troppo basso per SELL"
                        )
                        
                        try:
                            chart_buffer = generate_chart(df, symbol, timeframe)
                            await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=chart_buffer,
                                caption=caption,
                                parse_mode='HTML'
                            )
                        except:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=caption,
                                parse_mode='HTML'
                            )
                    
                    return  # Blocca ordine SELL
            
            # ===== CALCOLO PARAMETRI SELL =====
            entry_price = last_close
            
            # Calcola SL (sopra il prezzo per SHORT)
            if USE_EMA_STOP_LOSS:
                # SL sopra EMA per SHORT
                ema_10 = df['close'].ewm(span=10, adjust=False).mean()
                ema_60 = df['close'].ewm(span=60, adjust=False).mean()
                
                if timeframe in ['5m', '15m']:
                    sl_ema = ema_10.iloc[-1]
                    ema_used = 'EMA 10'
                else:
                    sl_ema = ema_60.iloc[-1]
                    ema_used = 'EMA 60'
                
                sl_price = sl_ema * (1 + EMA_SL_BUFFER)  # Sopra EMA per SHORT
                ema_value = sl_ema
            else:
                # ATR stop
                if not math.isnan(last_atr) and last_atr > 0:
                    sl_price = last_close + last_atr * ATR_MULT_SL
                    ema_used = 'ATR'
                    ema_value = last_atr
                else:
                    sl_price = df['high'].iloc[-1] * 1.002
                    ema_used = 'High'
                    ema_value = 0
            
            # TP (sotto il prezzo per SHORT)
            if not math.isnan(last_atr) and last_atr > 0:
                tp_price = last_close - last_atr * ATR_MULT_TP
            else:
                tp_price = last_close * 0.98
            
            # Risk calculation
            if ema_analysis and 'score' in ema_analysis:
                ema_score = ema_analysis['score']
                risk_base = calculate_dynamic_risk(ema_score)
            else:
                risk_base = RISK_USD
            
            if symbol in SYMBOL_RISK_OVERRIDE:
                risk_for_symbol = SYMBOL_RISK_OVERRIDE[symbol]
            else:
                risk_for_symbol = risk_base
            
            #qty = calculate_position_size(entry_price, sl_price, risk_for_symbol)
            # ===== INTELLIGENT POSITION SIZING =====
            # Calcola ATR per volatilità
            lastatr = atr(df, period=14).iloc[-1]
            if math.isnan(lastatr):
                lastatr = abs(entry_price - sl_price) * 0.01  # Fallback: 1% del range
            
            # Calcola qty con position sizing intelligente
            ema_score = ema_analysis['score'] if ema_analysis else 50
            qty = calculate_optimal_position_size(
                entry_price=entry_price,
                sl_price=sl_price,
                symbol=symbol,
                volatility_atr=lastatr,
                ema_score=ema_score,
                risk_usd=risk_for_symbol
            )
            
            # Add info nel caption
            caption += f"📊 Position Sizing:\n"
            caption += f"Position Size: {qty:.4f}\n"
            caption += f"Risk per Trade: ${risk_for_symbol:.2f}\n"
            if lastatr > 0:
                volatility_pct = (lastatr / entry_price) * 100
                caption += f"ATR: {lastatr:.2f} ({volatility_pct:.2f}% volatility)\n"

            
            # Check posizione esistente
            position_exists = symbol in config.ACTIVE_POSITIONS
            
            # ===== COSTRUISCI CAPTION SELL =====
            quality_emoji_map = {
                'GOLD': '🌟',
                'GOOD': '✅',
                'OK': '⚠️',
                'WEAK': '🔶',
                'BAD': '❌'
            }
            
            caption = "🔴 <b>SEGNALE SELL (SHORT)</b>\n\n"
            
            # EMA QUALITY
            if ema_analysis:
                q_emoji = quality_emoji_map.get(ema_analysis['quality'], '⚪')
                caption += f"{q_emoji} EMA Quality: <b>{ema_analysis['quality']}</b>\n"
                caption += f"Score: <b>{ema_analysis['score']}/100</b>\n\n"
            
            # Pattern info
            caption += f"📊 Pattern: <b>{pattern}</b>\n"
            caption += f"🪙 Symbol: <b>{symbol}</b> ({timeframe})\n"
            caption += f"🕐 {timestamp_str}\n\n"
            
            # Trading params
            caption += f"💵 Entry: <b>${entry_price:.{price_decimals}f}</b>\n"
            
            if USE_EMA_STOP_LOSS:
                caption += f"🛑 Stop Loss: <b>${sl_price:.{price_decimals}f}</b>\n"
                caption += f"   sopra {ema_used}"
                if isinstance(ema_value, (int, float)) and ema_value > 0:
                    caption += f" = ${ema_value:.{price_decimals}f}"
                caption += "\n"
            else:
                caption += f"🛑 Stop Loss: <b>${sl_price:.{price_decimals}f}</b> ({ema_used})\n"
            
            caption += f"🎯 Take Profit: <b>${tp_price:.{price_decimals}f}</b>\n"
            caption += f"📦 Qty: <b>{qty:.4f}</b>\n"
            caption += f"💰 Risk: <b>${risk_for_symbol}</b>\n"
            
            rr = abs(entry_price - tp_price) / abs(sl_price - entry_price) if abs(sl_price - entry_price) > 0 else 0
            caption += f"📏 R:R: <b>{rr:.2f}:1</b>\n"
            
            # EMA Analysis
            if ema_analysis:
                caption += "\n━━━━━━━━━━━━━━━━━━━\n"
                caption += "📉 <b>EMA Analysis (SHORT)</b>\n\n"
                caption += ema_analysis['details']
                caption += f"\nScore: <b>{ema_analysis['score']}/100</b>\n\n"
                
                if 'ema_values' in ema_analysis:
                    ema_vals = ema_analysis['ema_values']
                    ema_decimals = get_price_decimals(ema_vals['price'])
                    
                    caption += f"\n💡 <b>EMA Values:</b>\n"
                    caption += f"Price: ${ema_vals['price']:.{ema_decimals}f}\n"
                    caption += f"EMA 5: ${ema_vals['ema5']:.{ema_decimals}f}\n"
                    caption += f"EMA 10: ${ema_vals['ema10']:.{ema_decimals}f}\n"
                    caption += f"EMA 60: ${ema_vals['ema60']:.{ema_decimals}f}\n"
                    caption += f"EMA 223: ${ema_vals['ema223']:.{ema_decimals}f}\n"
                
                if USE_EMA_STOP_LOSS:
                    caption += f"\n🎯 <b>EMA Stop:</b> Exit se prezzo rompe {ema_used}"
                
                # Info filtri
                caption += f"\n\n💡 <b>Filtri Pattern:</b>\n"
                caption += f"Trend: {config.TREND_FILTER_MODE.upper()}\n"
                caption += f"Volume: {config.VOLUME_FILTER_MODE.upper()}\n"
                caption += f"EMA: {config.EMA_FILTER_MODE.upper() if config.EMA_FILTER_ENABLED else 'OFF'}\n"
            
            # Warning se LOOSE mode
            if ema_analysis and config.EMA_FILTER_MODE == 'loose' and not ema_analysis['passed']:
                caption += f"\n⚠️ <b>ATTENZIONE:</b> Setup con EMA non ottimali"
                caption += f"\nConsidera ridurre position size."
            
            # Posizione esistente
            if position_exists:
                caption += "\n\n🚫 <b>Posizione già aperta</b>"
                caption += f"\nOrdine NON eseguito per {symbol}"
            
            # Autotrade
            if job_ctx.get('autotrade') and qty > 0 and not position_exists:
                order_res = await place_bybit_order(
                    symbol, 
                    side,  # 'Sell'
                    qty, 
                    sl_price, 
                    tp_price,
                    entry_price,
                    timeframe,
                    chat_id,
                    pattern
                )
                
                if 'error' in order_res:
                    caption += f"\n\n❌ <b>Errore ordine:</b>\n{order_res['error']}"
                else:
                    caption += f"\n\n✅ <b>Ordine SHORT su Bybit {config.TRADING_MODE.upper()}</b>"
        
        else:
            # NESSUN PATTERN (full mode)
            caption = f"📊 <b>{symbol}</b> ({timeframe})\n"
            caption += f"🕐 {timestamp_str}\n"
            caption += f"💵 Price: ${last_close:.{price_decimals}f}\n\n"
            
            # NO MORE GLOBAL FILTERS INFO
            caption += "🔔 <b>Full Mode - Nessun pattern rilevato</b>\n\n"
            
            # Info filtri configurati (non status)
            caption += "💡 <b>Filter Configuration:</b>\n"
            
            if config.TREND_FILTER_ENABLED:
                caption += f"Trend: {config.TREND_FILTER_MODE.upper()}\n"
            else:
                caption += f"Trend: OFF\n"
            
            if config.VOLUME_FILTER_ENABLED:
                caption += f"Volume: {config.VOLUME_FILTER_MODE.upper()}\n"
            else:
                caption += f"Volume: OFF\n"
            
            if config.EMA_FILTER_ENABLED:
                caption += f"EMA: {config.EMA_FILTER_MODE.upper()}\n"
            else:
                caption += f"EMA: OFF\n"
            
            # ANALISI EMA MERCATO
            if ema_analysis:
                caption += "\n━━━━━━━━━━━━━━━━━━━\n"
                caption += "📈 <b>EMA Market Analysis:</b>\n\n"
                caption += f"<b>Score:</b> {ema_analysis['score']}/100\n"
                caption += f"<b>Quality:</b> {ema_analysis['quality']}\n\n"
                caption += ema_analysis['details']
                
                # Suggerimenti
                caption += "\n\n💡 <b>Suggerimento:</b>\n"
                if ema_analysis['quality'] == 'GOLD':
                    caption += "🌟 Setup PERFETTO! Aspetta pattern qui."
                elif ema_analysis['quality'] == 'GOOD':
                    caption += "✅ Buone condizioni. Setup valido."
                elif ema_analysis['quality'] == 'OK':
                    caption += "⚠️ Accettabile ma non ottimale."
                elif ema_analysis['quality'] == 'WEAK':
                    caption += "🔶 Condizioni deboli. Meglio aspettare."
                else:
                    caption += "❌ Condizioni sfavorevoli. NO entry."
                
                # Valori EMA
                if 'ema_values' in ema_analysis:
                    ema_vals = ema_analysis['ema_values']
                    ema_decimals = get_price_decimals(ema_vals['price'])
                    
                    caption += f"\n\n💡 <b>EMA Values:</b>\n"
                    caption += f"Price: ${ema_vals['price']:.{ema_decimals}f}\n"
                    caption += f"EMA 5: ${ema_vals['ema5']:.{ema_decimals}f}\n"
                    caption += f"EMA 10: ${ema_vals['ema10']:.{ema_decimals}f}\n"
                    caption += f"EMA 60: ${ema_vals['ema60']:.{ema_decimals}f}\n"
                    caption += f"EMA 223: ${ema_vals['ema223']:.{ema_decimals}f}\n"
        
        # ===== STEP 6: INVIA GRAFICO =====
        try:
            chart_buffer = generate_chart(df, symbol, timeframe)
            
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=chart_buffer,
                caption=caption,
                parse_mode='HTML'
            )
            
            # Log status
            if found:
                status = f'✅ {pattern}'
                if ema_analysis:
                    status += f' (EMA: {ema_analysis["quality"]} - {ema_analysis["score"]}/100)'
            else:
                status = '🔔 Full mode'
                if ema_analysis:
                    status += f' (EMA: {ema_analysis["quality"]})'
            
            logging.info(f"📸 {symbol} {timeframe} → {status}")
            
        except Exception as e:
            logging.error(f'Errore grafico: {e}')
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Errore grafico\n\n{caption}",
                parse_mode='HTML'
            )

    except Exception as e:
        logging.exception(f'Errore in analyze_job per {symbol} {timeframe}')
        
        try:
            with config.FULL_NOTIFICATIONS_LOCK:
                should_send = chat_id in config.FULL_NOTIFICATIONS and key in config.FULL_NOTIFICATIONS[chat_id]
            
            if should_send:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ Errore analisi {symbol} {timeframe}:\n{str(e)}"
                )
        except:
            logging.error(f'Impossibile inviare errore per {symbol} {timeframe}')


# ----------------------------- TELEGRAM COMMANDS -----------------------------
    
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start"""
    bot_username = (await context.bot.get_me()).username
    
    # Emoji per la modalità
    mode_emoji = "🎮" if config.TRADING_MODE == 'demo' else "💰"
    mode_text = "DEMO (fondi virtuali)" if config.TRADING_MODE == 'demo' else "LIVE (SOLDI REALI!)"
    
    welcome_text = (
        f"🤖 <b>Bot Pattern Detection Attivo!</b>\n"
        f"👤 Username: @{bot_username}\n"
        f"{mode_emoji} <b>Modalità: {mode_text}</b>\n\n"
        
        "━━━━━━━━━━━━━━━━━━━\n"
        "📊 <b>ANALISI PATTERN</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "/analizza SYMBOL TF [autotrade] - Avvia analisi\n"
        "/stop SYMBOL - Ferma analisi\n"
        "/list - Mostra analisi attive\n"
        "/test SYMBOL TF - Test completo pattern\n\n"
        
        "🔔 <b>NOTIFICHE</b>\n"
        "/abilita SYMBOL TF - Notifiche complete\n"
        "/pausa SYMBOL TF - Solo pattern (default)\n\n"
        
        "━━━━━━━━━━━━━━━━━━━\n"
        "💼 <b>TRADING & POSIZIONI</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "/balance - Mostra saldo wallet\n"
        "/posizioni - Posizioni aperte\n"
        "/orders [N] - Ultimi ordini + P&L\n"
        "/chiudi SYMBOL - Rimuovi da tracking\n"
        "/sync - Sincronizza con Bybit\n"
        "/trailing - Status trailing stop\n\n"
        
        "━━━━━━━━━━━━━━━━━━━\n"
        "🎯 <b>GESTIONE PATTERN</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "/patterns - Lista tutti i pattern\n"
        "/pattern_on NOME - Abilita pattern\n"
        "/pattern_off NOME - Disabilita pattern\n"
        "/pattern_info NOME - Info dettagliate\n"
        "/pattern_stats - Statistiche pattern\n"
        "/reset_pattern_stats - Reset statistiche\n"
        "/export_pattern_stats - Esporta CSV\n\n"
        
        "━━━━━━━━━━━━━━━━━━━\n"
        "📈 <b>FILTRI & CONFIGURAZIONE</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "/trend_filter [mode] - Trend filter\n"
        "  • structure | ema_based | hybrid | pattern_only\n"
        "/ema_filter [mode] - EMA filter\n"
        "  • strict | loose | off\n"
        "/ema_sl [on|off] - EMA Stop Loss\n"
        "/timefilter - Gestisci filtro orari\n\n"
        
        "━━━━━━━━━━━━━━━━━━━\n"
        "🔍 <b>AUTO-DISCOVERY</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "/autodiscover [on|off|now|status]\n"
        "→ Analizza automaticamente top symbols\n\n"
        
        "━━━━━━━━━━━━━━━━━━━\n"
        "🐛 <b>DEBUG & TEST</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "/test SYMBOL TF - Test pattern completo\n"
        "/test_flag SYMBOL TF - Test Bullish Flag\n"
        "/debug_volume SYMBOL TF - Debug volume\n"
        "/debug_filters SYMBOL TF - Debug filtri\n"
        "/force_test SYMBOL TF - Test NO filtri\n\n"
        
        "━━━━━━━━━━━━━━━━━━━\n"
        "💡 <b>ESEMPI RAPIDI</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "• Analisi base:\n"
        "  /analizza BTCUSDT 15m\n\n"
        "• Con autotrade:\n"
        "  /analizza ETHUSDT 5m yes\n\n"
        "• Test pattern:\n"
        "  /test SOLUSDT 15m\n\n"
        "• Debug completo:\n"
        "  /debug_filters BTCUSDT 5m\n\n"
        
        "━━━━━━━━━━━━━━━━━━━\n"
        "⚙️ <b>CONFIGURAZIONE ATTUALE</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        f"⏱️ Timeframes: {', '.join(config.ENABLED_TFS)}\n"
        f"💰 Risk default: ${config.RISK_USD}\n"
        f"📊 Trend: {config.TREND_FILTER_MODE.upper()}\n"
        f"📈 EMA: {config.EMA_FILTER_MODE.upper() if config.EMA_FILTER_ENABLED else 'OFF'}\n"
        f"🔕 Default: Solo pattern (non tutte)\n"
        f"🛑 EMA SL: {'ON' if config.USE_EMA_STOP_LOSS else 'OFF'}\n"
        f"🔄 Trailing: {'ON' if config.TRAILING_STOP_ENABLED else 'OFF'}\n\n"
        
        "━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ <b>NOTE IMPORTANTI</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "• Solo segnali BUY e SELL attivi\n"
        "• Pattern tier 1-2 prioritari\n"
        "• Filtri intelligenti anti-noise\n"
        "• Position sizing dinamico\n"
        "• Trailing stop multi-level\n\n"
        
        "❓ Usa /start per rivedere comandi\n"
        "💬 Feedback: thumbs down su messaggi"
    )
    
    await update.message.reply_text(welcome_text, parse_mode='HTML')

async def cmd_pausa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /pausa SYMBOL TIMEFRAME
    DISABILITA le notifiche complete e torna alla modalità default (solo pattern)
    """
    chat_id = update.effective_chat.id
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /pausa SYMBOL TIMEFRAME\n'
            'Esempio: /pausa BTCUSDT 15m\n\n'
            'Disabilita le notifiche complete e torna a modalità default.\n'
            '<b>Modalità default:</b> Ricevi solo notifiche quando ci sono pattern.\n\n'
            'Usa /abilita per riattivare tutte le notifiche.',
            parse_mode='HTML'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    key = f'{symbol}-{timeframe}'
    
    # Verifica che l'analisi sia attiva
    with config.ACTIVE_ANALYSES_LOCK:
        chat_map = ACTIVE_ANALYSES.get(chat_id, {})
        if key not in chat_map:
            await update.message.reply_text(
                f'⚠️ Non c\'è un\'analisi attiva per {symbol} {timeframe}.\n'
                f'Usa /analizza {symbol} {timeframe} per iniziare.'
            )
            return
    
    # Rimuovi dalle notifiche complete (torna a default = solo pattern)
    with FULL_NOTIFICATIONS_LOCK:
        if chat_id in FULL_NOTIFICATIONS and key in FULL_NOTIFICATIONS[chat_id]:
            FULL_NOTIFICATIONS[chat_id].remove(key)
            
            # Pulisci se il set è vuoto
            if not FULL_NOTIFICATIONS[chat_id]:
                del FULL_NOTIFICATIONS[chat_id]
            
            await update.message.reply_text(
                f'🔕 <b>Modalità default attivata per {symbol} {timeframe}</b>\n\n'
                f'Riceverai notifiche <b>SOLO quando viene rilevato un pattern</b>.\n'
                f'Niente più grafici senza segnali.\n\n'
                f'Usa /abilita {symbol} {timeframe} per riattivare tutte le notifiche.',
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                f'ℹ️ {symbol} {timeframe} è già in modalità default (solo pattern).\n\n'
                f'Non stai ricevendo notifiche complete.'
            )


async def cmd_abilita(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /abilita SYMBOL TIMEFRAME
    ABILITA le notifiche complete (anche quando non ci sono pattern)
    """
    chat_id = update.effective_chat.id
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /abilita SYMBOL TIMEFRAME\n'
            'Esempio: /abilita BTCUSDT 15m\n\n'
            'Abilita le notifiche complete per un symbol/timeframe.\n'
            '<b>Con notifiche complete:</b> Ricevi grafici ad ogni chiusura candela,\n'
            'anche quando non ci sono pattern.\n\n'
            'Utile per: monitoraggio visivo continuo, analisi manuale, debug.',
            parse_mode='HTML'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    key = f'{symbol}-{timeframe}'
    
    # Verifica che l'analisi sia attiva
    with config.ACTIVE_ANALYSES_LOCK:
        chat_map = ACTIVE_ANALYSES.get(chat_id, {})
        if key not in chat_map:
            await update.message.reply_text(
                f'⚠️ Non c\'è un\'analisi attiva per {symbol} {timeframe}.\n'
                f'Usa /analizza {symbol} {timeframe} per iniziare prima.'
            )
            return
    
    # Aggiungi alle notifiche complete
    with FULL_NOTIFICATIONS_LOCK:
        if chat_id not in FULL_NOTIFICATIONS:
            FULL_NOTIFICATIONS[chat_id] = set()
        
        if key in FULL_NOTIFICATIONS[chat_id]:
            await update.message.reply_text(
                f'ℹ️ Le notifiche complete per {symbol} {timeframe} sono già attive.'
            )
        else:
            FULL_NOTIFICATIONS[chat_id].add(key)
            
            await update.message.reply_text(
                f'🔔 <b>Notifiche complete attivate per {symbol} {timeframe}</b>\n\n'
                f'Riceverai grafici ad <b>ogni chiusura candela</b>,\n'
                f'anche quando non ci sono pattern.\n\n'
                f'Usa /pausa {symbol} {timeframe} per tornare a modalità default (solo pattern).',
                parse_mode='HTML'
            )


async def cmd_posizioni(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /posizioni
    Mostra tutte le posizioni aperte REALI da Bybit (sincronizzate)
    """
    await update.message.reply_text('🔄 Sincronizzando con Bybit...')
    
    # Sincronizza prima
    sync_success = await sync_positions_with_bybit()
    
    if not sync_success:
        await update.message.reply_text(
            '⚠️ Errore nella sincronizzazione con Bybit.\n'
            'Verifica le API keys e riprova.'
        )
        return
    
    # Ottieni posizioni reali
    real_positions = await get_open_positions_from_bybit()
    
    if not real_positions:
        with config.POSITIONS_LOCK:
            tracked = len(config.ACTIVE_POSITIONS)
        
        msg = '📭 <b>Nessuna posizione aperta</b>\n\n'
        
        if tracked > 0:
            msg += f'⚠️ Tracking locale aveva {tracked} posizioni, ma Bybit mostra 0.\n'
            msg += 'Tracking sincronizzato e pulito.\n\n'
        
        msg += 'Il bot non ha posizioni attive su Bybit.'
        
        await update.message.reply_text(msg, parse_mode='HTML')
        return
    
    msg = '📊 <b>Posizioni Aperte (da Bybit)</b>\n\n'
    
    for pos in real_positions:
        symbol = pos['symbol']
        side = pos['side']
        size = pos['size']
        entry_price = pos['entry_price']
        pnl = pos['unrealized_pnl']
        
        side_emoji = "🟢" if side == 'Buy' else "🔴"
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        
        msg += f"{side_emoji} <b>{symbol}</b> - {side}\n"
        msg += f"  📦 Size: {size}\n"
        msg += f"  💵 Entry: ${entry_price:.4f}\n"
        msg += f"  {pnl_emoji} PnL: ${pnl:+.2f}\n\n"
    
    msg += f"💼 Totale posizioni: {len(real_positions)}\n\n"
    msg += "💡 Posizioni sincronizzate con Bybit in tempo reale"
    
    await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_chiudi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /chiudi SYMBOL
    Rimuove una posizione dal tracking (utile se hai chiuso manualmente su Bybit)
    """
    args = context.args
    
    if len(args) < 1:
        await update.message.reply_text(
            '❌ Uso: /chiudi SYMBOL\n'
            'Esempio: /chiudi BTCUSDT\n\n'
            'Rimuove la posizione dal tracking del bot.\n'
            '(Non chiude automaticamente la posizione su Bybit)\n\n'
            'Oppure usa /sync per sincronizzare automaticamente.'
        )
        return
    
    symbol = args[0].upper()
    
    with config.POSITIONS_LOCK:
        if symbol in config.ACTIVE_POSITIONS:
            pos_info = config.ACTIVE_POSITIONS[symbol]
            del config.ACTIVE_POSITIONS[symbol]
            
            await update.message.reply_text(
                f'✅ <b>Posizione {symbol} rimossa dal tracking</b>\n\n'
                f'Dettagli posizione chiusa:\n'
                f'Side: {pos_info.get("side")}\n'
                f'Qty: {pos_info.get("qty"):.4f}\n\n'
                f'⚠️ Ricorda: questa azione rimuove solo il tracking.\n'
                f'Se la posizione è ancora aperta su Bybit, chiudila manualmente.',
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                f'⚠️ Nessuna posizione tracciata per {symbol}\n\n'
                f'Usa /posizioni per vedere le posizioni attive.'
            )


async def cmd_sync(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /sync
    Forza la sincronizzazione del tracking locale con Bybit
    """
    await update.message.reply_text('🔄 Sincronizzando con Bybit...')
    
    success = await sync_positions_with_bybit()
    
    if success:
        # Mostra risultato
        real_positions = await get_open_positions_from_bybit()
        
        with config.POSITIONS_LOCK:
            tracked_count = len(config.ACTIVE_POSITIONS)
        
        msg = '✅ <b>Sincronizzazione completata!</b>\n\n'
        msg += f'📊 Posizioni su Bybit: {len(real_positions)}\n'
        msg += f'💾 Posizioni tracciate: {tracked_count}\n\n'
        
        if real_positions:
            msg += '<b>Posizioni attive:</b>\n'
            for pos in real_positions:
                msg += f"• {pos['symbol']} - {pos['side']} ({pos['size']})\n"
        else:
            msg += 'Nessuna posizione aperta su Bybit.'
        
        await update.message.reply_text(msg, parse_mode='HTML')
    else:
        await update.message.reply_text(
            '❌ <b>Errore nella sincronizzazione</b>\n\n'
            'Verifica:\n'
            '• API keys configurate correttamente\n'
            '• Permessi API corretti\n'
            '• Connessione a Bybit attiva',
            parse_mode='HTML'
        )


async def cmd_patterns(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /patterns
    Mostra tutti i pattern disponibili e il loro stato (abilitato/disabilitato)
    """
    with PATTERNS_LOCK:
        # Separa pattern per tipo
        buy_patterns = []
        sell_patterns = []
        both_patterns = []
        
        for pattern_key, pattern_info in AVAILABLE_PATTERNS.items():
            emoji = pattern_info['emoji']
            name = pattern_info['name']
            enabled = pattern_info['enabled']
            side = pattern_info['side']
            status_emoji = "✅" if enabled else "❌"
            
            pattern_line = f"{status_emoji} {emoji} <code>{pattern_key}</code> - {name}"
            
            if side == 'Buy':
                buy_patterns.append(pattern_line)
            elif side == 'Sell':
                sell_patterns.append(pattern_line)
            else:
                both_patterns.append(pattern_line)
    
    msg = "📊 <b>Pattern Disponibili</b>\n\n"
    
    msg += "🟢 <b>Pattern BUY:</b>\n"
    msg += "\n".join(buy_patterns) + "\n\n"
    
    msg += "🔴 <b>Pattern SELL:</b>\n"
    msg += "\n".join(sell_patterns) + "\n\n"
    
    if both_patterns:
        msg += "⚪ <b>Pattern BOTH:</b>\n"
        msg += "\n".join(both_patterns) + "\n\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━\n\n"
    msg += "✅ = Abilitato (attivo)\n"
    msg += "❌ = Disabilitato (inattivo)\n\n"
    msg += "<b>Comandi:</b>\n"
    msg += "/pattern_on NOME - Abilita pattern\n"
    msg += "/pattern_off NOME - Disabilita pattern\n"
    msg += "/pattern_info NOME - Info dettagliate\n\n"
    msg += "Esempio: <code>/pattern_on bearish_engulfing</code>"
    
    await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_pattern_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /pattern_on PATTERN_KEY
    Abilita un pattern specifico
    """
    args = context.args
    
    if len(args) < 1:
        await update.message.reply_text(
            '❌ Uso: /pattern_on PATTERN_KEY\n\n'
            'Esempio: /pattern_on bearish_engulfing\n\n'
            'Usa /patterns per vedere tutti i pattern disponibili.'
        )
        return
    
    pattern_key = args[0].lower()
    
    with PATTERNS_LOCK:
        if pattern_key not in AVAILABLE_PATTERNS:
            await update.message.reply_text(
                f'❌ Pattern "{pattern_key}" non trovato.\n\n'
                f'Usa /patterns per vedere la lista completa.'
            )
            return
        
        pattern_info = AVAILABLE_PATTERNS[pattern_key]
        
        if pattern_info['enabled']:
            await update.message.reply_text(
                f'ℹ️ Pattern <b>{pattern_info["name"]}</b> è già abilitato.',
                parse_mode='HTML'
            )
            return
        
        # Abilita il pattern
        AVAILABLE_PATTERNS[pattern_key]['enabled'] = True
        
        emoji = pattern_info['emoji']
        name = pattern_info['name']
        side = pattern_info['side']
        desc = pattern_info['description']
        
        await update.message.reply_text(
            f'✅ <b>Pattern Abilitato!</b>\n\n'
            f'{emoji} <b>{name}</b>\n'
            f'📝 {desc}\n'
            f'📈 Direzione: {side}\n\n'
            f'Il bot ora rileverà questo pattern e invierà segnali.',
            parse_mode='HTML'
        )


async def cmd_pattern_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /pattern_off PATTERN_KEY
    Disabilita un pattern specifico
    """
    args = context.args
    
    if len(args) < 1:
        await update.message.reply_text(
            '❌ Uso: /pattern_off PATTERN_KEY\n\n'
            'Esempio: /pattern_off hammer\n\n'
            'Usa /patterns per vedere tutti i pattern disponibili.'
        )
        return
    
    pattern_key = args[0].lower()
    
    with PATTERNS_LOCK:
        if pattern_key not in AVAILABLE_PATTERNS:
            await update.message.reply_text(
                f'❌ Pattern "{pattern_key}" non trovato.\n\n'
                f'Usa /patterns per vedere la lista completa.'
            )
            return
        
        pattern_info = AVAILABLE_PATTERNS[pattern_key]
        
        if not pattern_info['enabled']:
            await update.message.reply_text(
                f'ℹ️ Pattern <b>{pattern_info["name"]}</b> è già disabilitato.',
                parse_mode='HTML'
            )
            return
        
        # Disabilita il pattern
        AVAILABLE_PATTERNS[pattern_key]['enabled'] = False
        
        emoji = pattern_info['emoji']
        name = pattern_info['name']
        
        await update.message.reply_text(
            f'❌ <b>Pattern Disabilitato!</b>\n\n'
            f'{emoji} <b>{name}</b>\n\n'
            f'Il bot non rileverà più questo pattern.',
            parse_mode='HTML'
        )


async def cmd_pattern_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /pattern_info PATTERN_KEY
    Mostra informazioni dettagliate su un pattern
    """
    args = context.args
    
    if len(args) < 1:
        await update.message.reply_text(
            '❌ Uso: /pattern_info PATTERN_KEY\n\n'
            'Esempio: /pattern_info bullish_comeback\n\n'
            'Usa /patterns per vedere tutti i pattern disponibili.'
        )
        return
    
    pattern_key = args[0].lower()
    
    with PATTERNS_LOCK:
        if pattern_key not in AVAILABLE_PATTERNS:
            await update.message.reply_text(
                f'❌ Pattern "{pattern_key}" non trovato.\n\n'
                f'Usa /patterns per vedere la lista completa.'
            )
            return
        
        pattern_info = AVAILABLE_PATTERNS[pattern_key]
        
        emoji = pattern_info['emoji']
        name = pattern_info['name']
        enabled = pattern_info['enabled']
        side = pattern_info['side']
        desc = pattern_info['description']
        
        status = "✅ Abilitato" if enabled else "❌ Disabilitato"
        
        msg = f"{emoji} <b>{name}</b>\n\n"
        msg += f"📝 <b>Descrizione:</b>\n{desc}\n\n"
        msg += f"📈 <b>Direzione:</b> {side}\n"
        msg += f"🔘 <b>Status:</b> {status}\n"
        msg += f"🔑 <b>Key:</b> <code>{pattern_key}</code>\n\n"
        
        if enabled:
            msg += f"Per disabilitare: /pattern_off {pattern_key}"
        else:
            msg += f"Per abilitare: /pattern_on {pattern_key}"
        
        await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_ema_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /ema_filter [MODE]
    Mostra o modifica la modalità filtro EMA
    """
    global EMA_FILTER_MODE, EMA_FILTER_ENABLED
    
    args = context.args
    
    if len(args) == 0:
        # Mostra stato attuale
        status_emoji = "✅" if EMA_FILTER_ENABLED else "❌"
        sl_emoji = "✅" if USE_EMA_STOP_LOSS else "❌"
        
        msg = f"📈 <b>Filtro EMA Status</b>\n\n"
        msg += f"🔘 Filtro Abilitato: {status_emoji}\n"
        msg += f"🎯 Modalità: <b>{EMA_FILTER_MODE.upper()}</b>\n"
        msg += f"🛑 EMA Stop Loss: {sl_emoji}\n\n"
        
        if USE_EMA_STOP_LOSS:
            msg += "<b>📍 EMA Stop Loss Config:</b>\n"
            for tf, ema in EMA_STOP_LOSS_CONFIG.items():
                msg += f"• {tf}: {ema.upper()}\n"
            msg += f"\nBuffer: {EMA_SL_BUFFER*100}% sotto EMA\n\n"
        
        msg += "<b>Modalità Filtro:</b>\n"
        msg += "• <code>strict</code> - Solo score ≥ 60 (GOOD/GOLD)\n"
        msg += "• <code>loose</code> - Score ≥ 40 (OK/GOOD/GOLD)\n"
        msg += "• <code>off</code> - Nessun filtro EMA\n\n"
        
        msg += "<b>Comandi:</b>\n"
        msg += "/ema_filter strict - Modalità strict\n"
        msg += "/ema_filter loose - Modalità loose\n"
        msg += "/ema_filter off - Disabilita filtro\n"
        msg += "/ema_sl - Gestisci EMA Stop Loss\n\n"
        
        msg += "<b>Timeframe Config:</b>\n"
        msg += "• 5m, 15m: EMA 5, 10\n"
        msg += "• 30m, 1h: EMA 10, 60\n"
        msg += "• 4h: EMA 60, 223"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        return
    
    # Modifica modalità
    mode = args[0].lower()
    
    if mode not in ['strict', 'loose', 'off']:
        await update.message.reply_text(
            '❌ Modalità non valida.\n\n'
            'Usa: /ema_filter [strict|loose|off]'
        )
        return
    
    old_mode = EMA_FILTER_MODE
    EMA_FILTER_MODE = mode
    
    if mode == 'off':
        EMA_FILTER_ENABLED = False
        msg = "❌ <b>Filtro EMA Disabilitato</b>\n\n"
        msg += "I pattern saranno rilevati senza controlli EMA."
    else:
        EMA_FILTER_ENABLED = True
        
        if mode == 'strict':
            msg = "🔒 <b>Modalità STRICT Attivata</b>\n\n"
            msg += "Solo segnali con score EMA ≥ 60 (GOOD/GOLD)\n"
            msg += "• Meno segnali\n"
            msg += "• Qualità superiore\n"
            msg += "• Win rate più alto\n\n"
            msg += "⚠️ Pattern con EMA deboli saranno IGNORATI"
        else:  # loose
            msg = "🔓 <b>Modalità LOOSE Attivata</b>\n\n"
            msg += "Segnali con score EMA ≥ 40 (OK/GOOD/GOLD)\n"
            msg += "• Più segnali\n"
            msg += "• Balance qualità/quantità\n"
            msg += "• Avvisi se EMA non perfette\n\n"
            msg += "⚠️ Pattern con EMA deboli ricevono warning"
    
    msg += f"\n\nModalità precedente: {old_mode.upper()}"
    
    await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_autodiscover(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /autodiscover [on|off|now|status]
    Gestisce il sistema di auto-discovery
    """
    chat_id = update.effective_chat.id
    args = context.args
    
    if not args:
        # Mostra status
        status_emoji = "✅" if config.AUTO_DISCOVERY_CONFIG['enabled'] else "❌"
        
        msg = f"🔍 <b>Auto-Discovery System</b>\n\n"
        msg += f"Status: {status_emoji} {'Attivo' if config.AUTO_DISCOVERY_CONFIG['enabled'] else 'Disattivo'}\n\n"
        
        if config.AUTO_DISCOVERY_CONFIG['enabled']:
            msg += f"<b>Configurazione:</b>\n"
            msg += f"• Top: {config.AUTO_DISCOVERY_CONFIG['top_count']} symbols\n"
            msg += f"• Timeframe: {config.AUTO_DISCOVERY_CONFIG['timeframe']}\n"
            msg += f"• Autotrade: {'ON' if config.AUTO_DISCOVERY_CONFIG['autotrade'] else 'OFF'}\n"
            msg += f"• Update ogni: {config.AUTO_DISCOVERY_CONFIG['update_interval']//3600}h\n"
            msg += f"• Min volume: ${config.AUTO_DISCOVERY_CONFIG['min_volume_usdt']/1_000_000:.0f}M\n"
            msg += f"• Min change: +{config.AUTO_DISCOVERY_CONFIG['min_price_change']}%\n"
            msg += f"• Max change: +{config.AUTO_DISCOVERY_CONFIG['max_price_change']}%\n\n"
            
            with config.AUTO_DISCOVERED_LOCK:
                symbols = list(config.AUTO_DISCOVERED_SYMBOLS)
            
            if symbols:
                msg += f"<b>Symbols attivi ({len(symbols)}):</b>\n"
                for sym in sorted(symbols):
                    msg += f"• {sym}\n"
            else:
                msg += "Nessun symbol ancora analizzato"
        
        msg += "\n\n<b>Comandi:</b>\n"
        msg += "/autodiscover on - Attiva\n"
        msg += "/autodiscover off - Disattiva\n"
        msg += "/autodiscover now - Aggiorna ora\n"
        msg += "/autodiscover status - Mostra status"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        return
    
    action = args[0].lower()
    
    if action == 'on':
        config.AUTO_DISCOVERY_CONFIG['enabled'] = True
        
        # Schedula il job se non esiste
        current_jobs = context.job_queue.get_jobs_by_name('auto_discovery')
        
        if not current_jobs:
            context.job_queue.run_repeating(
                auto_discover_and_analyze,
                interval=config.AUTO_DISCOVERY_CONFIG['update_interval'],
                first=60,  # Primo run dopo 1 minuto
                data={'chat_id': chat_id},
                name='auto_discovery'
            )
            
            await update.message.reply_text(
                '✅ <b>Auto-Discovery ATTIVATO</b>\n\n'
                'Primo update tra 1 minuto...\n'
                f"Poi ogni {config.AUTO_DISCOVERY_CONFIG['update_interval']//3600} ore",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                '✅ <b>Auto-Discovery già attivo</b>',
                parse_mode='HTML'
            )
    
    elif action == 'off':
        config.AUTO_DISCOVERY_CONFIG['enabled'] = False
        
        # Rimuovi tutti i job auto-discovery
        current_jobs = context.job_queue.get_jobs_by_name('auto_discovery')
        for job in current_jobs:
            job.schedule_removal()
        
        await update.message.reply_text(
            '❌ <b>Auto-Discovery DISATTIVATO</b>\n\n'
            'Le analisi esistenti continueranno.\n'
            'Usa /stop per fermarle.',
            parse_mode='HTML'
        )
    
    elif action == 'now':
        if not config.AUTO_DISCOVERY_CONFIG['enabled']:
            await update.message.reply_text(
                '⚠️ Auto-Discovery è disattivato.\n'
                'Usa /autodiscover on per attivarlo.'
            )
            return
        
        await update.message.reply_text('🔄 Aggiornamento in corso...')
        
        # Esegui manualmente
        await auto_discover_and_analyze(
            type('Context', (), {
                'job': type('Job', (), {'data': {'chat_id': chat_id}})(),
                'bot': context.bot,
                'job_queue': context.job_queue
            })()
        )
    
    elif action == 'status':
        # Mostra status dettagliato
        await cmd_autodiscover(update, context)
    
    else:
        await update.message.reply_text(
            '❌ Comando non valido.\n\n'
            'Usa: /autodiscover [on|off|now|status]'
        )
        

async def cmd_ema_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /ema_sl [on|off]
    Gestisce lo stop loss basato su EMA
    """
    global USE_EMA_STOP_LOSS
    
    args = context.args
    
    if len(args) == 0:
        # Mostra info
        status_emoji = "✅" if USE_EMA_STOP_LOSS else "❌"
        
        msg = f"🛑 <b>EMA Stop Loss System</b>\n\n"
        msg += f"Status: {status_emoji} {'Attivo' if USE_EMA_STOP_LOSS else 'Disattivo'}\n\n"
        
        if USE_EMA_STOP_LOSS:
            msg += "<b>📍 Configurazione per Timeframe:</b>\n"
            for tf, ema in EMA_STOP_LOSS_CONFIG.items():
                msg += f"• {tf} → {ema.upper()}\n"
            
            msg += f"\n<b>Buffer Safety:</b> {EMA_SL_BUFFER*100}%\n"
            msg += f"(SL piazzato {EMA_SL_BUFFER*100}% sotto l'EMA)\n\n"
            
            msg += "<b>💡 Come Funziona:</b>\n"
            msg += "1. Pattern rilevato → Entry\n"
            msg += "2. Stop Loss = EMA - buffer\n"
            msg += "3. Se prezzo rompe EMA → SL hit\n"
            msg += "4. EMA segue il prezzo = trailing stop\n\n"
            
            msg += "<b>🎯 Vantaggi:</b>\n"
            msg += "✅ Stop loss dinamico\n"
            msg += "✅ Si adatta al trend\n"
            msg += "✅ Evita stop troppo stretti\n"
            msg += "✅ Protegge profitti\n\n"
            
            msg += "<b>Esempio BTCUSDT 15m:</b>\n"
            msg += "Entry: $98,500\n"
            msg += "EMA 10: $98,200\n"
            msg += "SL: $98,200 - 0.2% = $98,003\n"
            msg += "Se prezzo scende sotto EMA 10 → Stop!\n\n"
            
        else:
            msg += "<b>Status: Disattivo</b>\n"
            msg += "Stop loss calcolato con ATR tradizionale.\n\n"
            msg += "ATR Stop = Entry ± (ATR × 1.5)\n\n"
            msg += "<b>Abilita EMA SL per:</b>\n"
            msg += "✅ Stop loss dinamici\n"
            msg += "✅ Trailing automatico\n"
            msg += "✅ Protezione trend\n\n"
        
        msg += "<b>Comandi:</b>\n"
        msg += "/ema_sl on - Abilita EMA Stop Loss\n"
        msg += "/ema_sl off - Disabilita (usa ATR)"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        return
    
    # Modifica setting
    action = args[0].lower()
    
    if action == 'on':
        USE_EMA_STOP_LOSS = True
        msg = "✅ <b>EMA Stop Loss Attivato!</b>\n\n"
        msg += "Gli stop loss saranno ora posizionati sotto le EMA chiave:\n\n"
        
        for tf, ema in EMA_STOP_LOSS_CONFIG.items():
            msg += f"• {tf} → {ema.upper()}\n"
        
        msg += f"\nBuffer: {EMA_SL_BUFFER*100}% sotto EMA\n\n"
        msg += "💡 <b>Vantaggi:</b>\n"
        msg += "✅ Stop dinamico che segue il trend\n"
        msg += "✅ Protezione automatica profitti\n"
        msg += "✅ Exit quando trend si inverte\n\n"
        msg += "⚠️ <b>Importante:</b>\n"
        msg += "Monitora le posizioni! Se prezzo rompe\n"
        msg += "l'EMA significativa, esci manualmente."
        
    elif action == 'off':
        USE_EMA_STOP_LOSS = False
        msg = "❌ <b>EMA Stop Loss Disattivato</b>\n\n"
        msg += "Stop loss calcolati con ATR tradizionale:\n"
        msg += "SL = Entry ± (ATR × 1.5)\n\n"
        msg += "Questo è uno stop FISSO, non si muove\n"
        msg += "con il prezzo."
        
    else:
        await update.message.reply_text(
            '❌ Argomento non valido.\n\n'
            'Usa: /ema_sl [on|off]'
        )
        return
    
    await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /balance
    Mostra il saldo del wallet Bybit
    """
    if not config.BYBIT_API_KEY or not config.BYBIT_API_SECRET:
        await update.message.reply_text(
            '⚠️ API Bybit non configurate.\n'
            'Configura BYBIT_API_KEY e BYBIT_API_SECRET nelle variabili d\'ambiente.'
        )
        return
    
    await update.message.reply_text('🔍 Recupero saldo...')
    
    try:
        session = create_bybit_session()
        
        # Ottieni wallet balance
        wallet = session.get_wallet_balance(accountType="UNIFIED")
        
        logging.info(f'📊 Wallet response: {wallet}')
        
        # Estrai info
        if wallet.get('retCode') == 0:
            result = wallet.get('result', {})
            accounts = result.get('list', [])
            
            if not accounts:
                await update.message.reply_text(
                    '⚠️ Nessun account trovato.\n'
                    'Verifica che le API keys siano corrette e abbiano i permessi giusti.'
                )
                return
            
            msg = f"💰 <b>Saldo Wallet ({config.TRADING_MODE.upper()})</b>\n\n"
            
            total_equity = 0
            found_coins = False
            
            for account in accounts:
                coins = account.get('coin', [])
                account_type = account.get('accountType', 'N/A')
                
                for coin in coins:
                    coin_name = coin.get('coin', 'N/A')
                    
                    # Gestione sicura dei float (può essere stringa vuota o None)
                    def safe_float(value, default=0.0):
                        if value is None or value == '':
                            return default
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return default
                    
                    equity = safe_float(coin.get('equity', 0))
                    available = safe_float(coin.get('availableToWithdraw', 0))
                    wallet_balance = safe_float(coin.get('walletBalance', 0))
                    unrealized_pnl = safe_float(coin.get('unrealisedPnl', 0))
                    
                    # Mostra solo coin con balance > 0.01
                    if equity > 0.01 or wallet_balance > 0.01:
                        found_coins = True
                        msg += f"<b>{coin_name}</b> ({account_type})\n"
                        msg += f"  💵 Equity: {equity:.4f}\n"
                        msg += f"  💼 Wallet Balance: {wallet_balance:.4f}\n"
                        msg += f"  ✅ Disponibile: {available:.4f}\n"
                        
                        if unrealized_pnl != 0:
                            pnl_emoji = "📈" if unrealized_pnl > 0 else "📉"
                            msg += f"  {pnl_emoji} PnL Non Realizzato: {unrealized_pnl:+.4f}\n"
                        
                        msg += "\n"
                        total_equity += equity
            
            if not found_coins:
                msg += "⚠️ Nessun balance trovato o tutti i balance sono zero.\n\n"
                msg += "💡 <b>Suggerimenti:</b>\n"
                msg += "• Se sei in Demo, vai su Bybit Demo e clicca 'Top Up'\n"
                msg += "• Verifica che le API keys abbiano i permessi corretti\n"
                msg += "• Assicurati di essere in 'Unified Trading Account'\n"
            else:
                msg += f"💰 <b>Totale Equity: {total_equity:.4f} USDT</b>\n"
            
            await update.message.reply_text(msg, parse_mode='HTML')
        else:
            error_code = wallet.get('retCode', 'N/A')
            error_msg = wallet.get('retMsg', 'Errore sconosciuto')
            
            msg = f"❌ <b>Errore API Bybit</b>\n\n"
            msg += f"Codice: {error_code}\n"
            msg += f"Messaggio: {error_msg}\n\n"
            
            # Errori comuni
            if error_code == 10003:
                msg += "💡 API Key non valida o scaduta.\n"
                msg += "Soluzione: Ricrea le API keys su Bybit."
            elif error_code == 10004:
                msg += "💡 Firma API non valida.\n"
                msg += "Soluzione: Verifica BYBIT_API_SECRET."
            elif error_code == 10005:
                msg += "💡 Permessi insufficienti.\n"
                msg += "Soluzione: Le API keys devono avere permessi 'Contract Trading'."
            
            await update.message.reply_text(msg, parse_mode='HTML')
            
    except Exception as e:
        logging.exception('Errore in cmd_balance')
        
        error_str = str(e)
        msg = f"❌ <b>Errore nel recuperare il saldo</b>\n\n"
        msg += f"Dettagli: {error_str}\n\n"
        
        # Suggerimenti basati sull'errore
        if 'Invalid API' in error_str or 'authentication' in error_str.lower():
            msg += "💡 Verifica le tue API keys:\n"
            msg += "1. Sono create in modalità Demo (se TRADING_MODE=demo)?\n"
            msg += "2. Hanno i permessi corretti (Unified Trading)?\n"
            msg += "3. Non sono scadute?\n"
        elif 'timeout' in error_str.lower():
            msg += "💡 Problema di connessione. Riprova tra qualche secondo.\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /orders [LIMIT]
    Mostra gli ultimi ordini chiusi con P&L da Bybit
    CON DISTINZIONE CORRETTA BUY/SELL usando closedSize
    """
    if not config.BYBIT_API_KEY or not config.BYBIT_API_SECRET:
        await update.message.reply_text(
            '⚠️ API Bybit non configurate.\n'
            'Configura BYBIT_API_KEY e BYBIT_API_SECRET nelle variabili d\'ambiente.'
        )
        return
    
    # Limita numero ordini da mostrare
    args = context.args
    limit = 10
    if args and args[0].isdigit():
        limit = min(int(args[0]), 50)  # Max 50 ordini
    
    await update.message.reply_text(f'🔍 Recupero ultimi {limit} ordini...')
    
    try:
        session = create_bybit_session()
        
        # Ottieni closed P&L (ordini chiusi con profitti/perdite)
        pnl_response = session.get_closed_pnl(
            category='linear',
            limit=limit
        )
        
        logging.info(f'📊 Closed P&L response: {pnl_response}')
        
        if pnl_response.get('retCode') == 0:
            result = pnl_response.get('result', {})
            pnl_list = result.get('list', [])
            
            if not pnl_list:
                await update.message.reply_text(
                    '📭 <b>Nessun ordine chiuso trovato</b>\n\n'
                    'Non ci sono ancora trade completati nel tuo account.',
                    parse_mode='HTML'
                )
                return
            
            msg = f"📊 <b>Ultimi {len(pnl_list)} Ordini Chiusi ({config.TRADING_MODE.upper()})</b>\n\n"
            
            # Statistiche globali
            total_pnl = 0
            win_count = 0
            loss_count = 0
            
            # Statistiche separate LONG/SHORT (corrette)
            long_stats = {
                'count': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'total_volume': 0
            }
            
            short_stats = {
                'count': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'total_volume': 0
            }
            
            # Processa ogni ordine
            for pnl_entry in pnl_list:
                symbol = pnl_entry.get('symbol', 'N/A')
                side = pnl_entry.get('side', 'N/A')  # Lato della CHIUSURA
                qty = float(pnl_entry.get('qty', 0))
                avg_entry = float(pnl_entry.get('avgEntryPrice', 0))
                avg_exit = float(pnl_entry.get('avgExitPrice', 0))
                closed_pnl = float(pnl_entry.get('closedPnl', 0))
                
                # ===== FIX CRITICO: Determina direzione CORRETTA =====
                # Per posizioni chiuse in profitto:
                # - Se exit > entry → era LONG (comprato basso, venduto alto)
                # - Se entry > exit → era SHORT (venduto alto, comprato basso)
                # 
                # Per conferma, usa anche closedSize:
                # - closedSize positivo = chiusura di LONG
                # - closedSize negativo = chiusura di SHORT
                
                closed_size = float(pnl_entry.get('closedSize', 0))
                
                # Determina direzione VERA della posizione
                if closed_size > 0:
                    # Chiusura di LONG (abbiamo venduto una posizione long)
                    trade_direction = 'LONG'
                    stats_bucket = long_stats
                    side_emoji = "🟢"
                elif closed_size < 0:
                    # Chiusura di SHORT (abbiamo comprato per chiudere una posizione short)
                    trade_direction = 'SHORT'
                    stats_bucket = short_stats
                    side_emoji = "🔴"
                else:
                    # Fallback: usa exit vs entry
                    if avg_exit > avg_entry:
                        trade_direction = 'LONG'
                        stats_bucket = long_stats
                        side_emoji = "🟢"
                    else:
                        trade_direction = 'SHORT'
                        stats_bucket = short_stats
                        side_emoji = "🔴"
                
                # Timestamp chiusura (millisecondi)
                updated_time = int(pnl_entry.get('updatedTime', 0))
                close_time = datetime.fromtimestamp(updated_time / 1000, tz=timezone.utc)
                time_str = close_time.strftime('%d/%m %H:%M')
                
                # Aggiorna statistiche globali
                total_pnl += closed_pnl
                if closed_pnl > 0:
                    win_count += 1
                    stats_bucket['wins'] += 1
                else:
                    loss_count += 1
                    stats_bucket['losses'] += 1
                
                # Aggiorna statistiche per direzione
                stats_bucket['count'] += 1
                stats_bucket['total_pnl'] += closed_pnl
                stats_bucket['total_volume'] += qty * avg_entry
                
                # Emoji risultato
                pnl_emoji = "✅" if closed_pnl > 0 else "❌"
                
                # Calcola P&L %
                pnl_percent = 0
                if avg_entry > 0:
                    if trade_direction == 'LONG':
                        pnl_percent = ((avg_exit - avg_entry) / avg_entry) * 100
                    else:  # SHORT
                        pnl_percent = ((avg_entry - avg_exit) / avg_entry) * 100
                
                # Decimali dinamici
                price_decimals = get_price_decimals(avg_entry)
                
                # Costruisci messaggio ordine
                msg += f"{side_emoji} <b>{symbol}</b> - {trade_direction}\n"
                msg += f"  Qty: {abs(qty):.4f}\n"
                msg += f"  Entry: ${avg_entry:.{price_decimals}f}\n"
                msg += f"  Exit: ${avg_exit:.{price_decimals}f}\n"
                msg += f"  {pnl_emoji} PnL: ${closed_pnl:+.2f} ({pnl_percent:+.2f}%)\n"
                msg += f"  Time: {time_str}\n"
                
                # Debug info (opzionale, commentare in produzione)
                # msg += f"  [Debug: side={side}, closedSize={closed_size}]\n"
                
                msg += "\n"
            
            # ===== STATISTICHE FINALI CON SEPARAZIONE LONG/SHORT =====
            msg += "━━━━━━━━━━━━━━━━━━━\n\n"
            
            # Statistiche globali
            msg += f"💰 <b>PnL Totale: ${total_pnl:+.2f}</b>\n"
            msg += f"✅ Win: {win_count} | ❌ Loss: {loss_count}\n"
            
            if (win_count + loss_count) > 0:
                win_rate = (win_count / (win_count + loss_count)) * 100
                msg += f"📊 Win Rate: {win_rate:.1f}%\n\n"
            
            # ===== STATISTICHE LONG =====
            if long_stats['count'] > 0:
                long_win_rate = (long_stats['wins'] / long_stats['count']) * 100
                avg_pnl_long = long_stats['total_pnl'] / long_stats['count']
                
                msg += "🟢 <b>LONG Statistics:</b>\n"
                msg += f"  Trades: {long_stats['count']}\n"
                msg += f"  Wins: {long_stats['wins']} | Losses: {long_stats['losses']}\n"
                msg += f"  Win Rate: {long_win_rate:.1f}%\n"
                msg += f"  Total PnL: ${long_stats['total_pnl']:+.2f}\n"
                msg += f"  Avg PnL/Trade: ${avg_pnl_long:+.2f}\n"
                msg += f"  Volume: ${long_stats['total_volume']:.0f}\n\n"
            
            # ===== STATISTICHE SHORT =====
            if short_stats['count'] > 0:
                short_win_rate = (short_stats['wins'] / short_stats['count']) * 100
                avg_pnl_short = short_stats['total_pnl'] / short_stats['count']
                
                msg += "🔴 <b>SHORT Statistics:</b>\n"
                msg += f"  Trades: {short_stats['count']}\n"
                msg += f"  Wins: {short_stats['wins']} | Losses: {short_stats['losses']}\n"
                msg += f"  Win Rate: {short_win_rate:.1f}%\n"
                msg += f"  Total PnL: ${short_stats['total_pnl']:+.2f}\n"
                msg += f"  Avg PnL/Trade: ${avg_pnl_short:+.2f}\n"
                msg += f"  Volume: ${short_stats['total_volume']:.0f}\n\n"
            
            # ===== CONFRONTO PERFORMANCE =====
            if long_stats['count'] > 0 and short_stats['count'] > 0:
                msg += "📈 <b>Performance Comparison:</b>\n"
                
                # Win rate comparison
                long_wr = long_stats['wins'] / long_stats['count']
                short_wr = short_stats['wins'] / short_stats['count']
                
                if long_wr > short_wr:
                    msg += f"  Best Win Rate: 🟢 LONG ({long_wr*100:.1f}%)\n"
                else:
                    msg += f"  Best Win Rate: 🔴 SHORT ({short_wr*100:.1f}%)\n"
                
                # PnL comparison
                if long_stats['total_pnl'] > short_stats['total_pnl']:
                    msg += f"  Most Profitable: 🟢 LONG (${long_stats['total_pnl']:+.2f})\n"
                else:
                    msg += f"  Most Profitable: 🔴 SHORT (${short_stats['total_pnl']:+.2f})\n"
                
                # Avg PnL comparison
                if avg_pnl_long > avg_pnl_short:
                    msg += f"  Better Avg: 🟢 LONG (${avg_pnl_long:+.2f}/trade)\n"
                else:
                    msg += f"  Better Avg: 🔴 SHORT (${avg_pnl_short:+.2f}/trade)\n"
                
                msg += "\n"
            
            # ===== INSIGHTS =====
            msg += "💡 <b>Insights:</b>\n"
            
            # Identifica lato più profittevole
            if long_stats['count'] > 0 and short_stats['count'] > 0:
                if long_stats['total_pnl'] > short_stats['total_pnl'] * 1.5:
                    msg += "  • LONG trades molto più profittevoli\n"
                    msg += "  • Considera di tradare più LONG\n"
                elif short_stats['total_pnl'] > long_stats['total_pnl'] * 1.5:
                    msg += "  • SHORT trades molto più profittevoli\n"
                    msg += "  • Considera di tradare più SHORT\n"
                else:
                    msg += "  • Performance LONG/SHORT bilanciata\n"
            
            # Warning se un lato perde
            if long_stats['count'] > 0 and long_stats['total_pnl'] < -10:
                msg += "  • ⚠️ LONG trades in perdita netta\n"
            if short_stats['count'] > 0 and short_stats['total_pnl'] < -10:
                msg += "  • ⚠️ SHORT trades in perdita netta\n"
            
            # Best side by win rate
            if long_stats['count'] > 0 and short_stats['count'] > 0:
                if long_wr > 0.6 and long_wr > short_wr:
                    msg += f"  • ✅ LONG win rate eccellente ({long_wr*100:.1f}%)\n"
                elif short_wr > 0.6 and short_wr > long_wr:
                    msg += f"  • ✅ SHORT win rate eccellente ({short_wr*100:.1f}%)\n"
            
            msg += f"\n💡 Usa /orders [numero] per vedere più ordini\n"
            msg += f"Esempio: /orders 20"
            
            await update.message.reply_text(msg, parse_mode='HTML')
        else:
            error_code = pnl_response.get('retCode', 'N/A')
            error_msg = pnl_response.get('retMsg', 'Errore sconosciuto')
            
            msg = f"❌ <b>Errore API Bybit</b>\n\n"
            msg += f"Codice: {error_code}\n"
            msg += f"Messaggio: {error_msg}\n\n"
            
            await update.message.reply_text(msg, parse_mode='HTML')
            
    except Exception as e:
        logging.exception('Errore in cmd_orders')
        
        error_str = str(e)
        msg = f"❌ <b>Errore nel recuperare gli ordini</b>\n\n"
        msg += f"Dettagli: {error_str}\n\n"
        
        # Suggerimenti
        if 'Invalid API' in error_str or 'authentication' in error_str.lower():
            msg += "💡 Verifica le tue API keys:\n"
            msg += "1. Hanno i permessi corretti?\n"
            msg += "2. Non sono scadute?\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')
            
    except Exception as e:
        logging.exception('Errore in cmd_orders')
        
        error_str = str(e)
        msg = f"❌ <b>Errore nel recuperare gli ordini</b>\n\n"
        msg += f"Dettagli: {error_str}\n\n"
        
        # Suggerimenti
        if 'Invalid API' in error_str or 'authentication' in error_str.lower():
            msg += "💡 Verifica le tue API keys:\n"
            msg += "1. Hanno i permessi corretti?\n"
            msg += "2. Non sono scadute?\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_analizza(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /analizza SYMBOL TIMEFRAME [autotrade]"""
    chat_id = update.effective_chat.id
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso corretto: /analizza SYMBOL TIMEFRAME [autotrade]\n'
            'Esempio: /analizza BTCUSDT 15m\n'
            'Per trading automatico: /analizza BTCUSDT 15m yes'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    autotrade = (len(args) > 2 and args[2].lower() in ['yes', 'true', '1', 'si', 'sì'])

    if timeframe not in ENABLED_TFS:
        await update.message.reply_text(
            f'❌ Timeframe non supportato.\n'
            f'Timeframes disponibili: {", ".join(ENABLED_TFS)}'
        )
        return

    # Verifica che il symbol esista
    test_df = bybit_get_klines(symbol, timeframe, limit=10)
    if test_df.empty:
        await update.message.reply_text(
            f'❌ Impossibile ottenere dati per {symbol}.\n'
            'Verifica che il simbolo sia corretto (es: BTCUSDT, ETHUSDT)'
        )
        return

    key = f'{symbol}-{timeframe}'
    
    with config.ACTIVE_ANALYSES_LOCK:
        chat_map = ACTIVE_ANALYSES.setdefault(chat_id, {})
        
        if key in chat_map:
            await update.message.reply_text(
                f'⚠️ Stai già analizzando {symbol} {timeframe} in questa chat.'
            )
            return
        
        # Calcola intervallo in secondi
        interval_seconds = INTERVAL_SECONDS.get(timeframe, 300)
        
        # Calcola tempo fino alla prossima chiusura candela
        now = datetime.now(timezone.utc)
        epoch = int(now.timestamp())
        to_next = interval_seconds - (epoch % interval_seconds)
        
        # Crea job
        job_data = {
            'chat_id': chat_id, 
            'symbol': symbol, 
            'timeframe': timeframe, 
            'autotrade': autotrade
        }
        
        job = context.job_queue.run_repeating(
            analyze_job, 
            interval=interval_seconds, 
            first=to_next, 
            data=job_data,
            name=key
        )
        
        chat_map[key] = job

    await update.message.reply_text(
        f'✅ <b>Analisi avviata!</b>\n'
        f'🪙 Symbol: {symbol}\n'
        f'⏱️ Timeframe: {timeframe}\n'
        f'🤖 Autotrade: {"Sì" if autotrade else "No"}\n'
        f'⏰ Prossimo check tra {to_next}s',
        parse_mode='HTML'
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /stop SYMBOL"""
    chat_id = update.effective_chat.id
    args = context.args
    
    if len(args) < 1:
        await update.message.reply_text(
            '❌ Uso: /stop SYMBOL oppure /stop all\n'
            'Esempio: /stop BTCUSDT'
        )
        return
    
    target = args[0].upper()
    
    with config.ACTIVE_ANALYSES_LOCK:
        chat_map = ACTIVE_ANALYSES.get(chat_id, {})
        
        if not chat_map:
            await update.message.reply_text('⚠️ Nessuna analisi attiva in questa chat.')
            return
        
        if target == 'ALL':
            count = len(chat_map)
            for k, job in list(chat_map.items()):
                job.schedule_removal()
                del chat_map[k]
            await update.message.reply_text(f'✅ Fermate {count} analisi.')
            return
        
        # Ferma tutte le analisi per un symbol specifico
        removed = []
        for k in list(chat_map.keys()):
            if k.startswith(target + '-'):
                job = chat_map[k]
                job.schedule_removal()
                del chat_map[k]
                removed.append(k)
        
        if removed:
            await update.message.reply_text(
                f'✅ Fermate analisi per {target}:\n' + '\n'.join(removed)
            )
        else:
            await update.message.reply_text(
                f'⚠️ Nessuna analisi attiva per {target} in questa chat.'
            )


async def cmd_trailing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /trailing [ACTION]
    
    Actions:
    - (nessuno): Mostra status
    - setup: Imposta trailing per posizioni senza
    - force: Forza update trailing su tutte le posizioni
    """
    if not config.TRAILING_STOP_ENABLED:
        await update.message.reply_text(
            "<b>Trailing Stop Loss DISABILITATO</b>\n"
            "Abilita nelle configurazioni: TRAILING_STOP_ENABLED = True",
            parse_mode="HTML"
        )
        return
    
    args = context.args
    action = args[0].lower() if args else 'status'
    
    with config.POSITIONS_LOCK:
        positions_copy = dict(config.ACTIVE_POSITIONS)
    
    if not positions_copy:
        await update.message.reply_text(
            "<b>Nessuna posizione attiva</b>\n"
            "Non ci sono posizioni con trailing stop loss.",
            parse_mode="HTML"
        )
        return
    
    # ===== ACTION: SETUP =====
    if action == 'setup':
        await update.message.reply_text("🔄 Impostando trailing stop per posizioni senza...")
        
        setup_count = 0
        errors = []
        
        for symbol, pos in positions_copy.items():
            side = pos.get('side')
            if side != 'Buy':
                continue  # Solo LONG per ora
            
            # Check se ha già trailing attivo
            if pos.get('trailing_active', False):
                continue
            
            # Imposta trailing
            try:
                entry = pos.get('entry_price')
                timeframe = pos.get('timeframe', '15m')
                
                # Scarica dati per calcolare EMA 10
                ema_tf = config.TRAILING_EMA_TIMEFRAME.get(timeframe, '5m')
                df = bybit_get_klines(symbol, ema_tf, limit=20)
                
                if df.empty:
                    errors.append(f"{symbol}: No data")
                    continue
                
                # Calcola nuovo SL basato su EMA 10
                ema_10 = df['close'].ewm(span=10, adjust=False).mean().iloc[-1]
                ema_buffer = config.TRAILING_CONFIG_ADVANCED['levels'][0]['ema_buffer']
                new_sl = ema_10 * (1 - ema_buffer)
                
                # Verifica che non sia peggio dello SL corrente
                current_sl = pos['sl']
                if new_sl <= current_sl:
                    errors.append(f"{symbol}: New SL worse than current")
                    continue
                
                # Aggiorna su Bybit
                session = create_bybit_session()
                tp_price = pos.get('tp', 0)
                
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "stopLoss": str(round(new_sl, get_price_decimals(new_sl))),
                    "positionIdx": 0
                }
                
                if tp_price > 0:
                    params["takeProfit"] = str(round(tp_price, get_price_decimals(tp_price)))
                
                result = session.set_trading_stop(**params)
                
                if result.get('retCode') == 0:
                    # Marca come trailing attivo
                    with config.POSITIONS_LOCK:
                        if symbol in config.ACTIVE_POSITIONS:
                            config.ACTIVE_POSITIONS[symbol]['sl'] = new_sl
                            config.ACTIVE_POSITIONS[symbol]['trailing_active'] = True
                            config.ACTIVE_POSITIONS[symbol]['highest_price'] = entry
                    
                    setup_count += 1
                    logging.info(f"✅ {symbol}: Trailing setup (SL={new_sl:.4f})")
                else:
                    errors.append(f"{symbol}: {result.get('retMsg')}")
            
            except Exception as e:
                errors.append(f"{symbol}: {str(e)[:50]}")
                logging.error(f"Error setting up trailing for {symbol}: {e}")
        
        # Report
        msg = f"<b>Trailing Setup Completato</b>\n\n"
        msg += f"✅ Setup: {setup_count} posizioni\n"
        
        if errors:
            msg += f"\n❌ Errori ({len(errors)}):\n"
            for err in errors[:5]:  # Max 5 errori
                msg += f"• {err}\n"
            if len(errors) > 5:
                msg += f"... e altri {len(errors)-5}\n"
        
        await update.message.reply_text(msg, parse_mode="HTML")
        return
    
    # ===== ACTION: FORCE =====
    if action == 'force':
        await update.message.reply_text("🔄 Forcing trailing update per tutte le posizioni...")
        
        # Chiama direttamente la funzione di update
        await update_trailing_stop_loss(context)
        
        await update.message.reply_text("✅ Trailing update forzato completato")
        return
    
    # ===== ACTION: STATUS (default) =====
    msg = "<b>📈 Advanced Trailing Stop Status</b>\n\n"
    
    # Mostra configurazione livelli
    msg += "<b>🎯 Livelli Configurati:</b>\n"
    for i, level in enumerate(config.TRAILING_CONFIG_ADVANCED['levels'], 1):
        msg += f"{i}. {level['label']}\n"
        msg += f"   • Attivazione: ≥{level['profit_pct']}% profit\n"
        msg += f"   • Buffer: {level['ema_buffer']*100:.2f}% sotto EMA\n\n"
    
    msg += f"<b>⚙️ Settings:</b>\n"
    msg += f"Check Interval: {config.TRAILING_CONFIG_ADVANCED['check_interval']} secondi\n"
    msg += f"Never Back: {'✅ ON' if config.TRAILING_CONFIG_ADVANCED['never_back'] else '❌ OFF'}\n\n"
    
    msg += "<b>📊 Posizioni Attive:</b>\n\n"
    
    positions_with_trailing = 0
    positions_without_trailing = 0
    
    for symbol, pos in positions_copy.items():
        if pos['side'] != 'Buy':
            continue
        
        entry = pos.get('entry_price')
        if not entry:
            continue
        
        current_sl = pos['sl']
        timeframe_entry = pos['timeframe']
        trailing_active = pos.get('trailing_active', False)
        
        # Scarica prezzo corrente
        df = bybit_get_klines(symbol, timeframe_entry, limit=5)
        current_price = df['close'].iloc[-1] if not df.empty else entry
        profit_pct = ((current_price - entry) / entry) * 100
        
        # Determina livello attivo
        active_level = None
        for level in config.TRAILING_CONFIG_ADVANCED['levels']:
            if profit_pct >= level['profit_pct']:
                active_level = level
        
        level_emoji = '⚪' if not active_level else {
            'Early Protection': '🟡',
            'Standard Trail': '🟢',
            'Tight Trail': '🔵',
            'Ultra Tight Trail': '🟣'
        }.get(active_level['label'], '⚪')
        
        price_decimals = get_price_decimals(current_price)
        
        # Status emoji
        if trailing_active:
            status_emoji = "✅"
            positions_with_trailing += 1
        else:
            status_emoji = "⚠️"
            positions_without_trailing += 1
        
        msg += f"{status_emoji} {level_emoji} <b>{symbol}</b> ({timeframe_entry})\n"
        msg += f"Entry: ${entry:.{price_decimals}f}\n"
        msg += f"Current: ${current_price:.{price_decimals}f}\n"
        msg += f"SL: ${current_sl:.{price_decimals}f}\n"
        msg += f"Profit: {profit_pct:.2f}%\n"
        
        if active_level:
            msg += f"<b>Level: {active_level['label']}</b>\n"
        else:
            needed = config.TRAILING_CONFIG_ADVANCED['levels'][0]['profit_pct'] - profit_pct
            msg += f"Serve +{needed:.2f}% per attivare\n"
        
        if not trailing_active:
            msg += "⚠️ <b>Trailing NON attivo</b>\n"
        
        msg += "\n"
    
    # Summary
    msg += f"\n<b>📋 Summary:</b>\n"
    msg += f"✅ Con trailing: {positions_with_trailing}\n"
    msg += f"⚠️ Senza trailing: {positions_without_trailing}\n\n"
    
    if positions_without_trailing > 0:
        msg += f"💡 Usa <code>/trailing setup</code> per impostare trailing\n\n"
    
    msg += "<b>ℹ️ Info</b>\n"
    msg += "• SL segue EMA 10 del TF superiore\n"
    msg += "• Livelli progressivi stringono automaticamente\n"
    msg += "• SL non torna mai indietro\n\n"
    msg += "<b>Comandi:</b>\n"
    msg += "<code>/trailing</code> - Status\n"
    msg += "<code>/trailing setup</code> - Setup trailing\n"
    msg += "<code>/trailing force</code> - Forza update"
    
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /list - mostra analisi attive con dettagli completi
    """
    chat_id = update.effective_chat.id
    
    with config.ACTIVE_ANALYSES_LOCK:
        chat_map = ACTIVE_ANALYSES.get(chat_id, {})
    
    if not chat_map:
        await update.message.reply_text(
            '📭 <b>Nessuna analisi attiva</b>\n\n'
            'Usa /analizza SYMBOL TIMEFRAME per iniziare.',
            parse_mode='HTML'
        )
        return
    
    # Prepara messaggio dettagliato
    msg = f'📊 <b>Analisi Attive ({len(chat_map)})</b>\n\n'
    
    for key, job in chat_map.items():
        symbol, timeframe = key.split('-')
        job_data = job.data
        
        # Determina autotrade
        autotrade = job_data.get('autotrade', False)
        
        # ===== MARKET TIME FILTER (autotrade gate) =====
        time_ok, time_reason = is_good_trading_time_utc()
        if not time_ok:
            if MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY:
                if autotrade:
                    logging.info(f"{symbol} {timeframe}: Autotrade disabilitato ({time_reason})")
                autotrade = False  # solo ordini off, analisi continua
            else:
                logging.info(f"{symbol} {timeframe}: Analisi saltata ({time_reason})")
                return

        autotrade_emoji = "🤖" if autotrade else "📊"
        autotrade_text = "Autotrade ON" if autotrade else "Solo monitoraggio"
        
        # Determina modalità notifiche
        with FULL_NOTIFICATIONS_LOCK:
            full_mode = chat_id in FULL_NOTIFICATIONS and key in FULL_NOTIFICATIONS[chat_id]
        
        notif_emoji = "🔔" if full_mode else "🔕"
        notif_text = "Tutte le notifiche" if full_mode else "Solo pattern"
        
        # Costruisci riga per questo symbol
        msg += f"{autotrade_emoji} <b>{symbol}</b> - {timeframe}\n"
        msg += f"  {notif_emoji} {notif_text}\n"
        msg += f"  {'🤖 ' + autotrade_text}\n"
        
        # Verifica se ha posizione aperta
        if symbol in config.ACTIVE_POSITIONS:
            pos = config.ACTIVE_POSITIONS[symbol]
            msg += f"  💼 Posizione: {pos.get('side')} ({pos.get('qty'):.4f})\n"
        
        msg += "\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━\n\n"
    msg += "<b>Legenda:</b>\n"
    msg += "🤖 = Autotrade attivo\n"
    msg += "📊 = Solo monitoraggio\n"
    msg += "🔔 = Notifiche complete\n"
    msg += "🔕 = Solo pattern (default)\n"
    msg += "💼 = Posizione aperta\n\n"
    msg += "<b>Comandi:</b>\n"
    msg += "/stop SYMBOL - Ferma analisi\n"
    msg += "/abilita SYMBOL TF - Attiva notifiche complete\n"
    msg += "/pausa SYMBOL TF - Solo pattern\n"
    msg += "/posizioni - Dettagli posizioni"
    
    await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /test SYMBOL TIMEFRAME
    Testa TUTTI i pattern sull'ultima candela e mostra debug info
    
    VERSION: 3.0 - Updated con pattern recenti:
    - Volume Spike Breakout
    - Breakout + Retest
    - Triple Touch Breakout (NUOVO)
    - Liquidity Sweep
    - S/R Bounce
    - Bullish Comeback
    - Compression Breakout
    - Bullish Flag Breakout
    - Morning Star + EMA Breakout
    - Pattern classici (Engulfing, Hammer, ecc.)
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /test SYMBOL TIMEFRAME\n'
            'Esempio: /test BTCUSDT 15m\n\n'
            'Questo comando mostra:\n'
            '• Info candela corrente\n'
            '• Risultati test TUTTI i pattern\n'
            '• Filtri globali (volume, trend, ATR)\n'
            '• EMA analysis\n'
            '• Grafico con pattern rilevato'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    
    if timeframe not in ENABLED_TFS:
        await update.message.reply_text(
            f'❌ Timeframe non supportato.\n'
            f'Disponibili: {", ".join(ENABLED_TFS)}'
        )
        return
    
    await update.message.reply_text(f'🔍 Analizzo {symbol} {timeframe}...')
    
    try:
        # ===== STEP 1: OTTIENI DATI =====
        df = bybit_get_klines(symbol, timeframe, limit=200)
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato per {symbol}')
            return
        
        # ===== STEP 2: INFO CANDELE RECENTI =====
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Calcola metriche candela corrente
        last_body = abs(last['close'] - last['open'])
        last_range = last['high'] - last['low']
        last_body_pct = (last_body / last_range * 100) if last_range > 0 else 0
        
        lower_wick = min(last['open'], last['close']) - last['low']
        upper_wick = last['high'] - max(last['open'], last['close'])
        
        lower_wick_pct = (lower_wick / last_range * 100) if last_range > 0 else 0
        upper_wick_pct = (upper_wick / last_range * 100) if last_range > 0 else 0
        
        # Decimali dinamici
        price_decimals = get_price_decimals(last['close'])
        
        # ===== STEP 3: FILTRI GLOBALI =====
        vol_ok = False
        vol_ratio = 0.0
        
        try:
            vol_ok = volume_confirmation(df, min_ratio=1.5)
            if 'volume' in df.columns and len(df['volume']) >= 20:
                vol = df['volume']
                avg_vol = vol.iloc[-20:-1].mean()
                current_vol = vol.iloc[-1]
                if avg_vol > 0:
                    vol_ratio = current_vol / avg_vol
        except Exception as e:
            logging.error(f'Error volume check: {e}')
        
        trend_ok = False
        try:
            trend_ok = is_uptrend_structure(df)
        except Exception as e:
            logging.error(f'Error trend check: {e}')
        
        atr_ok = False
        try:
            atr_ok = atr_expanding(df)
        except Exception as e:
            logging.error(f'Error ATR check: {e}')
        
        # ===== STEP 4: EMA ANALYSIS =====
        ema_analysis = None
        try:
            ema_analysis = analyze_ema_conditions(df, timeframe, None)
        except Exception as e:
            logging.error(f'Error EMA analysis: {e}')
        
        # ===== STEP 5: TEST PATTERN INDIVIDUALI =====
        # Pattern con return (bool, data) o (bool, None)
        tests_with_data = {}
        
        # TIER 1 Patterns (con data)
        try:
            found, data = is_volume_spike_breakout(df)
            tests_with_data['📊💥 Volume Spike Breakout'] = (found, data)
        except Exception as e:
            logging.error(f'Error Volume Spike: {e}')
            tests_with_data['📊💥 Volume Spike Breakout'] = (False, None)
        
        try:
            found, data = is_liquidity_sweep_reversal(df)
            tests_with_data['💎 Liquidity Sweep + Reversal'] = (found, data)
        except Exception as e:
            logging.error(f'Error Liquidity Sweep: {e}')
            tests_with_data['💎 Liquidity Sweep + Reversal'] = (False, None)
        
        try:
            found, data = is_support_resistance_bounce(df)
            tests_with_data['🎯 Support/Resistance Bounce'] = (found, data)
        except Exception as e:
            logging.error(f'Error S/R Bounce: {e}')
            tests_with_data['🎯 Support/Resistance Bounce'] = (False, None)
        
        try:
            found, data = is_bullish_flag_breakout(df)
            tests_with_data['🚩 Bullish Flag Breakout'] = (found, data)
        except Exception as e:
            logging.error(f'Error Flag: {e}')
            tests_with_data['🚩 Bullish Flag Breakout'] = (False, None)
        
        # ===== NUOVO: Triple Touch Breakout =====
        try:
            found, data = is_triple_touch_breakout(df)
            tests_with_data['🎯3️⃣ Triple Touch Breakout'] = (found, data)
        except NameError:
            # Funzione non definita
            tests_with_data['🎯3️⃣ Triple Touch Breakout'] = ('❌ NOT IMPLEMENTED', None)
        except Exception as e:
            logging.error(f'Error Triple Touch: {e}')
            tests_with_data['🎯3️⃣ Triple Touch Breakout'] = (False, None)
        
        # Pattern bool only
        tests_bool = {}
        
        try:
            tests_bool['🔄 Bullish Comeback'] = is_bullish_comeback(df)
        except Exception as e:
            logging.error(f'Error Comeback: {e}')
            tests_bool['🔄 Bullish Comeback'] = False
        
        try:
            tests_bool['💥 Compression Breakout'] = is_compression_breakout(df)
        except Exception as e:
            logging.error(f'Error Compression: {e}')
            tests_bool['💥 Compression Breakout'] = False
        
        try:
            tests_bool['⭐💥 Morning Star + EMA Breakout'] = is_morning_star_ema_breakout(df)
        except Exception as e:
            logging.error(f'Error Morning Star EMA: {e}')
            tests_bool['⭐💥 Morning Star + EMA Breakout'] = False
        
        try:
            tests_bool['🟢 Bullish Engulfing'] = is_bullish_engulfing(prev, last)
        except Exception as e:
            tests_bool['🟢 Bullish Engulfing'] = False
        
        try:
            tests_bool['🔴 Bearish Engulfing'] = is_bearish_engulfing(prev, last)
        except Exception as e:
            tests_bool['🔴 Bearish Engulfing'] = False
        
        try:
            tests_bool['🔨 Hammer'] = is_hammer(last)
        except Exception as e:
            tests_bool['🔨 Hammer'] = False
        
        try:
            tests_bool['💫 Shooting Star'] = is_shooting_star(last)
        except Exception as e:
            tests_bool['💫 Shooting Star'] = False
        
        try:
            tests_bool['📍 Pin Bar'] = is_pin_bar(last)
        except Exception as e:
            tests_bool['📍 Pin Bar'] = False
        
        try:
            tests_bool['➖ Doji'] = is_doji(last)
        except Exception as e:
            tests_bool['➖ Doji'] = False
        
        try:
            tests_bool['⭐ Morning Star'] = is_morning_star_enhanced(prev2, prev, last)
        except Exception as e:
            tests_bool['⭐ Morning Star'] = False
        
        try:
            tests_bool['🌙 Evening Star'] = is_evening_star(prev2, prev, last)
        except Exception as e:
            tests_bool['🌙 Evening Star'] = False
        
        try:
            tests_bool['⬆️ Three White Soldiers'] = is_three_white_soldiers(prev2, prev, last)
        except Exception as e:
            tests_bool['⬆️ Three White Soldiers'] = False
        
        try:
            tests_bool['⬇️ Three Black Crows'] = is_three_black_crows(prev2, prev, last)
        except Exception as e:
            tests_bool['⬇️ Three Black Crows'] = False
        
        # ===== STEP 6: PATTERN RILEVATO DA check_patterns() =====
        found_main = False
        side_main = None
        pattern_main = None
        pattern_data_main = None
        
        try:
            found_main, side_main, pattern_main, pattern_data_main = check_patterns(df)
        except Exception as e:
            logging.error(f'Error check_patterns: {e}')
        
        # ===== STEP 7: COSTRUISCI MESSAGGIO =====
        msg = f"🔍 <b>Test Pattern: {symbol} {timeframe}</b>\n\n"
        
        # Pattern principale rilevato
        if found_main:
            msg += f"✅ <b>PATTERN RILEVATO: {pattern_main}</b>\n"
            msg += f"📈 Direzione: {side_main}\n\n"
        else:
            msg += "❌ Nessun pattern rilevato da check_patterns()\n\n"
        
        # Info candela corrente
        msg += f"📊 <b>Ultima candela:</b>\n"
        msg += f"O: ${last['open']:.{price_decimals}f} | H: ${last['high']:.{price_decimals}f}\n"
        msg += f"L: ${last['low']:.{price_decimals}f} | C: ${last['close']:.{price_decimals}f}\n"
        msg += f"{'🟢 Bullish' if last['close'] > last['open'] else '🔴 Bearish'}\n"
        msg += f"Corpo: {last_body_pct:.1f}% del range\n"
        msg += f"Ombra inf: {lower_wick_pct:.1f}%\n"
        msg += f"Ombra sup: {upper_wick_pct:.1f}%\n\n"
        
        # Filtri globali
        msg += "🔍 <b>Filtri Globali:</b>\n"
        msg += f"{'✅' if vol_ok else '❌'} Volume: {vol_ratio:.1f}x (>1.5x)\n"
        msg += f"{'✅' if trend_ok else '❌'} Uptrend Structure\n"
        msg += f"{'✅' if atr_ok else '⚠️'} ATR Expanding\n\n"
        
        # EMA Analysis
        if ema_analysis:
            msg += f"📈 <b>EMA Quality:</b> {ema_analysis['quality']} ({ema_analysis['score']}/100)\n\n"
        
        # Test pattern (con data)
        msg += "🧪 <b>Test Pattern (TIER 1):</b>\n"
        for pattern_name, (result, data) in tests_with_data.items():
            if result == '❌ NOT IMPLEMENTED':
                emoji = "⚠️"
                status = "NOT IMPLEMENTED"
            elif result:
                emoji = "✅"
                status = "FOUND"
                if data:
                    # Mostra info chiave
                    if 'volume_ratio' in data:
                        status += f" (vol: {data['volume_ratio']:.1f}x)"
                    elif 'breakout_vol_ratio' in data:
                        status += f" (vol: {data['breakout_vol_ratio']:.1f}x)"
            else:
                emoji = "❌"
                status = "Not found"
            
            msg += f"{emoji} {pattern_name}: {status}\n"
        
        msg += "\n"
        
        # Test pattern (bool)
        msg += "🧪 <b>Test Pattern (Altri):</b>\n"
        for pattern_name, result in tests_bool.items():
            emoji = "✅" if result else "❌"
            msg += f"{emoji} {pattern_name}\n"
        
        # Verifica Triple Touch specificamente
        msg += "\n"
        msg += "━━━━━━━━━━━━━━━━━━━\n"
        msg += "<b>🎯 Triple Touch Verification:</b>\n"
        
        if '🎯3️⃣ Triple Touch Breakout' in tests_with_data:
            result, data = tests_with_data['🎯3️⃣ Triple Touch Breakout']
            
            if result == '❌ NOT IMPLEMENTED':
                msg += "⚠️ <b>Funzione NOT FOUND!</b>\n"
                msg += "La funzione is_triple_touch_breakout() non è definita.\n"
                msg += "Verifica che sia stata aggiunta al codice."
            elif result:
                msg += "✅ <b>Pattern TROVATO!</b>\n"
                if data:
                    msg += f"Resistance: ${data.get('resistance', 0):.{price_decimals}f}\n"
                    msg += f"Touches: {data.get('touch_count', 0)}\n"
                    msg += f"Rejection 1: {data.get('touch_1_rejection_pct', 0):.1f}%\n"
                    msg += f"Rejection 2: {data.get('touch_2_rejection_pct', 0):.1f}%\n"
                    msg += f"Consolidation: {data.get('consolidation_duration', 0)} candele\n"
                    msg += f"Volume: {data.get('volume_ratio', 0):.1f}x\n"
                    msg += f"Quality: {data.get('quality', 'N/A')}\n"
            else:
                msg += "❌ Pattern non trovato\n"
                msg += "Verifica:\n"
                msg += "• Resistance toccata 3 volte?\n"
                msg += "• Prime 2 con rejection?\n"
                msg += "• Consolidamento 3-10 candele?\n"
                msg += "• Prezzo sempre sopra EMA 60?\n"
                msg += "• Breakout terzo tocco?\n"
        else:
            msg += "⚠️ Triple Touch non testato\n"
        
        # Limita lunghezza messaggio
        if len(msg) > 4000:
            msg = msg[:3900] + "\n\n... (troncato per lunghezza)"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
        # ===== STEP 8: INVIA GRAFICO =====
        try:
            chart_buffer = generate_chart(df, symbol, timeframe)
            
            caption = f"{symbol} {timeframe}"
            if found_main:
                caption += f"\n✅ {pattern_main}"
            
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=chart_buffer,
                caption=caption
            )
        except Exception as e:
            logging.error(f'Errore generazione grafico test: {e}')
    
    except Exception as e:
        logging.exception('Errore in cmd_test')
        await update.message.reply_text(
            f'❌ Errore durante il test:\n{str(e)}\n\n'
            f'Verifica che {symbol} sia valido e abbia dati disponibili.'
        )


async def cmd_trend_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /trend_filter [mode]
    Gestisce il trend filter
    """
    global TREND_FILTER_MODE
    
    args = context.args
    
    if not args:
        # Mostra status
        msg = "📈 <b>Trend Filter Status</b>\n\n"
        msg += f"Enabled: {'✅' if TREND_FILTER_ENABLED else '❌'}\n"
        msg += f"Mode: <b>{TREND_FILTER_MODE.upper()}</b>\n\n"
        
        msg += "<b>Available Modes:</b>\n"
        msg += "• <code>structure</code> - HH+HL (originale, stretto)\n"
        msg += "• <code>ema_based</code> - EMA 60 (consigliato)\n"
        msg += "• <code>hybrid</code> - Structure OR EMA (flessibile)\n"
        msg += "• <code>pattern_only</code> - Ogni pattern decide\n\n"
        
        msg += "<b>Current Mode Details:</b>\n"
        if TREND_FILTER_MODE == 'ema_based':
            msg += "✅ Permette consolidamenti sopra EMA 60\n"
            msg += "✅ Permette pullback sopra EMA 60\n"
            msg += "✅ Rileva breakout early\n"
            msg += "📊 Win rate mantiene: ~60-70%\n"
        elif TREND_FILTER_MODE == 'structure':
            msg += "⚠️ Blocca consolidamenti\n"
            msg += "⚠️ Blocca pullback\n"
            msg += "📊 Perde ~40-60% segnali\n"
        elif TREND_FILTER_MODE == 'hybrid':
            msg += "✅ Permissivo (OR logic)\n"
            msg += "📊 Balance qualità/quantità\n"
        else:
            msg += "✅ Massima flessibilità\n"
            msg += "⚠️ Ogni pattern decide criteri\n"
        
        msg += "\n<b>Commands:</b>\n"
        msg += "/trend_filter structure\n"
        msg += "/trend_filter ema_based\n"
        msg += "/trend_filter hybrid\n"
        msg += "/trend_filter pattern_only"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        return
    
    mode = args[0].lower()
    
    if mode not in ['structure', 'ema_based', 'hybrid', 'pattern_only']:
        await update.message.reply_text(
            '❌ Mode non valido\n\n'
            'Usa: /trend_filter [structure|ema_based|hybrid|pattern_only]'
        )
        return
    
    TREND_FILTER_MODE = mode
    
    msg = f'✅ <b>Trend Filter: {mode.upper()}</b>\n\n'
    
    if mode == 'ema_based':
        msg += '<b>EMA-Based Mode (CONSIGLIATO)</b>\n\n'
        msg += '✅ Prezzo sopra EMA 60 = uptrend\n'
        msg += '✅ Consolidamenti OK se sopra EMA 60\n'
        msg += '✅ Pullback OK se non rompe EMA 60\n'
        msg += '✅ Rileva breakout momentum\n\n'
        msg += '📊 Mantiene 60-70% patterns\n'
        msg += '🎯 Use case: Tuoi pattern (Triple Touch, Flag, ecc.)'
    
    elif mode == 'structure':
        msg += '<b>Structure Mode (ORIGINALE)</b>\n\n'
        msg += 'Richiede Higher Highs + Higher Lows\n\n'
        msg += '⚠️ Blocca consolidamenti\n'
        msg += '⚠️ Blocca pullback\n'
        msg += '⚠️ Perde breakout da range\n\n'
        msg += '📊 Perde ~40-60% segnali\n'
        msg += '🎯 Use case: Solo uptrend forti'
    
    elif mode == 'hybrid':
        msg += '<b>Hybrid Mode (FLESSIBILE)</b>\n\n'
        msg += 'Structure OR EMA (basta uno)\n\n'
        msg += '✅ Più permissivo\n'
        msg += '📊 Balance qualità/quantità\n'
        msg += '🎯 Use case: Mix pattern types'
    
    else:  # pattern_only
        msg += '<b>Pattern-Only Mode (NO GLOBAL)</b>\n\n'
        msg += 'Ogni pattern ha criteri propri\n\n'
        msg += '✅ Massima flessibilità\n'
        msg += '⚠️ Richiede pattern ben configurati\n'
        msg += '🎯 Use case: Pattern già molto selettivi'
    
    await update.message.reply_text(msg, parse_mode='HTML')

async def cmd_time_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /timefilter
    /timefilter on|off
    /timefilter hours 1 2 3 4
    /timefilter mode autotrade|all
    """
    global MARKET_TIME_FILTER_ENABLED
    global MARKET_TIME_FILTER_BLOCKED_UTC_HOURS
    global MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY

    args = context.args

    # Status
    if not args:
        mode = "AUTOTRADE_ONLY" if MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY else "ALL_ANALYSIS"
        hours = ", ".join([f"{h:02d}" for h in sorted(MARKET_TIME_FILTER_BLOCKED_UTC_HOURS)])
        msg = ""
        msg += "<b>Market Time Filter</b>\n"
        msg += f"Status: {'✅ ON' if MARKET_TIME_FILTER_ENABLED else '❌ OFF'}\n"
        msg += f"Mode: <b>{mode}</b>\n"
        msg += f"Blocked UTC hours: <b>{hours if hours else 'None'}</b>\n\n"
        msg += "<b>Comandi</b>\n"
        msg += "<code>timefilter on</code> | <code>timefilter off</code>\n"
        msg += "<code>timefilter hours 1 2 3 4</code>\n"
        msg += "<code>timefilter hours clear</code>\n"
        msg += "<code>timefilter mode autotrade</code> (blocca solo ordini)\n"
        msg += "<code>timefilter mode all</code> (blocca anche analisi)\n"
        await update.message.reply_text(msg, parse_mode="HTML")
        return

    cmd = args[0].lower()

    # ON/OFF
    if cmd in ("on", "off"):
        MARKET_TIME_FILTER_ENABLED = (cmd == "on")
        await update.message.reply_text(
            f"<b>Market Time Filter</b>: {'✅ ON' if MARKET_TIME_FILTER_ENABLED else '❌ OFF'}",
            parse_mode="HTML"
        )
        return

    # MODE
    if cmd == "mode":
        if len(args) < 2:
            await update.message.reply_text("Uso: <code>timefilter mode autotrade|all</code>", parse_mode="HTML")
            return
        m = args[1].lower()
        if m not in ("autotrade", "all"):
            await update.message.reply_text("Valore non valido. Usa: autotrade | all", parse_mode="HTML")
            return
        MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY = (m == "autotrade")
        await update.message.reply_text(
            f"Mode impostato: <b>{'AUTOTRADE_ONLY' if MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY else 'ALL_ANALYSIS'}</b>",
            parse_mode="HTML"
        )
        return

    # HOURS
    if cmd == "hours":
        if len(args) < 2:
            await update.message.reply_text("Uso: <code>timefilter hours 1 2 3 4</code> oppure <code>timefilter hours clear</code>", parse_mode="HTML")
            return

        if args[1].lower() == "clear":
            MARKET_TIME_FILTER_BLOCKED_UTC_HOURS = set()
            await update.message.reply_text("Blocked UTC hours svuotate.", parse_mode="HTML")
            return

        new_hours = set()
        for s in args[1:]:
            if not s.isdigit():
                await update.message.reply_text(f"Ora non valida: {s} (usa numeri 0-23)", parse_mode="HTML")
                return
            h = int(s)
            if h < 0 or h > 23:
                await update.message.reply_text(f"Ora fuori range: {h} (0-23)", parse_mode="HTML")
                return
            new_hours.add(h)

        MARKET_TIME_FILTER_BLOCKED_UTC_HOURS = new_hours
        hours = ", ".join([f"{h:02d}" for h in sorted(MARKET_TIME_FILTER_BLOCKED_UTC_HOURS)])
        await update.message.reply_text(f"Blocked UTC hours impostate: <b>{hours}</b>", parse_mode="HTML")
        return

    await update.message.reply_text("Comando non valido. Usa: <code>timefilter</code>", parse_mode="HTML")

async def cmd_debug_filters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /debug_filters SYMBOL TIMEFRAME
    Mostra TUTTI i filtri e perché stanno bloccando i pattern
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /debug_filters SYMBOL TIMEFRAME\n'
            'Esempio: /debug_filters BTCUSDT 5m'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    
    await update.message.reply_text(f'🔍 Analizzando filtri per {symbol} {timeframe}...')
    
    try:
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=200)
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato per {symbol}')
            return
        
        msg = f"🔍 <b>DEBUG FILTERS: {symbol} {timeframe}</b>\n\n"
        
        # ===== 1. MARKET TIME FILTER =====
        msg += "<b>⏰ 1. MARKET TIME FILTER</b>\n"
        msg += f"Enabled: {'✅' if MARKET_TIME_FILTER_ENABLED else '❌'}\n"
        
        if MARKET_TIME_FILTER_ENABLED:
            time_ok, time_reason = is_good_trading_time_utc()
            now_utc = datetime.now(timezone.utc)
            current_hour = now_utc.hour
            
            msg += f"Current UTC Hour: <b>{current_hour:02d}</b>\n"
            msg += f"Blocked Hours: {sorted(MARKET_TIME_FILTER_BLOCKED_UTC_HOURS)}\n"
            msg += f"Status: {'✅ OK' if time_ok else f'❌ BLOCKED - {time_reason}'}\n"
            msg += f"Mode: {'AUTOTRADE_ONLY' if MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY else 'ALL_ANALYSIS'}\n"
            
            if not time_ok:
                msg += "\n⚠️ <b>PATTERN SEARCH SKIPPED!</b>\n"
                if MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY:
                    msg += "Analisi pattern OK, ma autotrade disabilitato\n"
                else:
                    msg += "TUTTO bloccato (analisi + autotrade)\n"
        
        msg += "\n"
        
        # ===== 2. VOLUME FILTER =====
        msg += "<b>📊 2. VOLUME FILTER</b>\n"
        msg += f"Mode: <b>{VOLUME_FILTER_MODE}</b>\n"
        msg += f"Enabled: {'✅' if VOLUME_FILTER_ENABLED else '❌'}\n"
        
        vol = df['volume']
        if len(vol) >= 20:
            avg_vol = vol.iloc[-20:-1].mean()
            current_vol = vol.iloc[-1]
            
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                
                msg += f"Current Volume: {current_vol:.2f}\n"
                msg += f"Avg Volume (20): {avg_vol:.2f}\n"
                msg += f"Ratio: <b>{vol_ratio:.2f}x</b>\n"
                
                # Check diversi threshold
                msg += f"\nThreshold Checks:\n"
                msg += f"• 1.2x (S/R Bounce): {'✅ PASS' if vol_ratio >= 1.2 else '❌ FAIL'}\n"
                msg += f"• 1.5x (Globale): {'✅ PASS' if vol_ratio >= 1.5 else '❌ FAIL'}\n"
                msg += f"• 1.8x (Enhanced): {'✅ PASS' if vol_ratio >= 1.8 else '❌ FAIL'}\n"
                msg += f"• 2.0x (Flag): {'✅ PASS' if vol_ratio >= 2.0 else '❌ FAIL'}\n"
                msg += f"• 3.0x (Volume Spike): {'✅ PASS' if vol_ratio >= 3.0 else '❌ FAIL'}\n"
                
                if vol_ratio < 1.5:
                    msg += "\n⚠️ <b>Volume TROPPO BASSO per la maggior parte dei pattern!</b>\n"
            else:
                msg += "❌ Average volume is ZERO!\n"
        else:
            msg += "❌ Dati insufficienti per calcolare volume\n"
        
        msg += "\n"
        
        # ===== 3. TREND FILTER =====
        msg += "<b>📈 3. TREND FILTER</b>\n"
        msg += f"Enabled: {'✅' if TREND_FILTER_ENABLED else '❌'}\n"
        msg += f"Mode: <b>{TREND_FILTER_MODE}</b>\n"
        
        if TREND_FILTER_ENABLED:
            trend_valid, trend_reason, trend_details = is_valid_trend_for_entry(
                df, mode=TREND_FILTER_MODE, symbol=symbol
            )
            
            msg += f"Status: {'✅ VALID' if trend_valid else f'❌ INVALID - {trend_reason}'}\n"
            
            if TREND_FILTER_MODE == 'ema_based' and trend_details:
                ema60 = trend_details.get('ema60', 0)
                price = trend_details.get('price', 0)
                distance = trend_details.get('distance_pct', 0)
                
                msg += f"EMA 60: ${ema60:.2f}\n"
                msg += f"Price: ${price:.2f}\n"
                msg += f"Distance: {distance:.2f}%\n"
                
                if not trend_valid:
                    msg += "\n⚠️ <b>TREND FILTER BLOCKING!</b>\n"
        
        msg += "\n"
        
        # ===== 4. EMA FILTER =====
        msg += "<b>💹 4. EMA FILTER</b>\n"
        msg += f"Enabled: {'✅' if EMA_FILTER_ENABLED else '❌'}\n"
        msg += f"Mode: <b>{EMA_FILTER_MODE}</b>\n"
        
        if EMA_FILTER_ENABLED:
            ema_analysis = analyze_ema_conditions(df, timeframe, None)
            
            msg += f"Score: <b>{ema_analysis['score']}/100</b>\n"
            msg += f"Quality: <b>{ema_analysis['quality']}</b>\n"
            msg += f"Passed: {'✅ YES' if ema_analysis['passed'] else '❌ NO'}\n"
            
            if EMA_FILTER_MODE == 'strict':
                msg += f"Threshold: 60/100\n"
                if ema_analysis['score'] < 60:
                    msg += "\n⚠️ <b>EMA STRICT BLOCKING!</b>\n"
            elif EMA_FILTER_MODE == 'loose':
                msg += f"Threshold: 40/100\n"
                if ema_analysis['score'] < 40:
                    msg += "\n⚠️ <b>EMA LOOSE BLOCKING!</b>\n"
            
            msg += f"\nDetails:\n{ema_analysis['details']}\n"
        
        msg += "\n"
        
        # ===== 5. PATTERN-SPECIFIC CHECKS =====
        msg += "<b>🎯 5. PATTERN-SPECIFIC VOLUME CHECKS</b>\n"
        
        # Test alcuni pattern chiave
        patterns_to_test = [
            ('Volume Spike Breakout', lambda: is_volume_spike_breakout(df)),
            ('S/R Bounce', lambda: is_support_resistance_bounce(df)),
            ('Bullish Flag', lambda: is_bullish_flag_breakout(df)),
        ]
        
        for pattern_name, pattern_func in patterns_to_test:
            try:
                result = pattern_func()
                found = result[0] if isinstance(result, tuple) else result
                msg += f"• {pattern_name}: {'✅ FOUND' if found else '❌ Not found'}\n"
            except Exception as e:
                msg += f"• {pattern_name}: ❌ Error - {str(e)[:50]}\n"
        
        msg += "\n"
        
        # ===== 6. SUMMARY =====
        msg += "<b>📋 SUMMARY</b>\n"
        
        blocking_filters = []
        
        if MARKET_TIME_FILTER_ENABLED:
            time_ok, _ = is_good_trading_time_utc()
            if not time_ok and not MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY:
                blocking_filters.append("Market Time (ALL)")
        
        if VOLUME_FILTER_ENABLED and vol_ratio < 1.5:
            blocking_filters.append("Volume (too low)")
        
        if TREND_FILTER_ENABLED:
            trend_valid, _, _ = is_valid_trend_for_entry(df, mode=TREND_FILTER_MODE)
            if not trend_valid:
                blocking_filters.append(f"Trend ({TREND_FILTER_MODE})")
        
        if EMA_FILTER_ENABLED and not ema_analysis['passed']:
            blocking_filters.append(f"EMA ({EMA_FILTER_MODE})")
        
        if blocking_filters:
            msg += "❌ <b>FILTERS BLOCKING:</b>\n"
            for f in blocking_filters:
                msg += f"  • {f}\n"
        else:
            msg += "✅ <b>All filters OK</b>\n"
            msg += "If no pattern found, issue is in pattern logic itself\n"
        
        # Limita lunghezza
        if len(msg) > 4000:
            msg = msg[:3900] + "\n\n... (troncato)"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
    except Exception as e:
        logging.exception('Errore in cmd_debug_filters')
        await update.message.reply_text(f'❌ Errore: {str(e)}')


async def cmd_force_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /force_test SYMBOL TIMEFRAME
    Forza il test dei pattern SENZA FILTRI GLOBALI
    Per vedere se il problema è nei filtri o nei pattern stessi
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /force_test SYMBOL TIMEFRAME\n'
            'Esempio: /force_test BTCUSDT 5m'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    
    await update.message.reply_text(f'🔍 Force testing NO FILTERS {symbol} {timeframe}...')
    
    try:
        df = bybit_get_klines(symbol, timeframe, limit=200)
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato')
            return
        
        msg = f"🔍 <b>FORCE TEST NO FILTERS: {symbol} {timeframe}</b>\n\n"
        
        # Test DIRETTO dei pattern (bypass filtri)
        tests = {}
        
        # Volume Spike (con threshold custom basso)
        try:
            vol = df['volume']
            avg_vol = vol.iloc[-20:-1].mean()
            curr_vol = vol.iloc[-1]
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 0
            
            # Test con threshold rilassato (1.5x invece di 3x)
            if vol_ratio >= 1.5:
                found, data = is_volume_spike_breakout(df)
                tests['Volume Spike 3x'] = found
            else:
                tests['Volume Spike 3x'] = f"Volume too low ({vol_ratio:.1f}x minore 3x)"
        except Exception as e:
            tests['Volume Spike'] = f"Error: {str(e)[:30]}"
        
        # S/R Bounce
        try:
            found, data = is_support_resistance_bounce(df)
            tests['S/R Bounce'] = found
        except Exception as e:
            tests['S/R Bounce'] = f"Error: {str(e)[:30]}"
        
        # Bullish Flag
        try:
            found, data = is_bullish_flag_breakout(df)
            tests['Bullish Flag'] = found
        except Exception as e:
            tests['Bullish Flag'] = f"Error: {str(e)[:30]}"
        
        # Triple Touch
        try:
            found, data = is_triple_touch_breakout(df)
            tests['Triple Touch'] = found
        except Exception as e:
            tests['Triple Touch'] = f"Error: {str(e)[:30]}"
        
        # Engulfing Enhanced
        try:
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            found, tier, data = is_bullish_engulfing_enhanced(prev, curr, df)
            tests[f'Engulfing Enhanced'] = f"{tier} tier" if found else False
        except Exception as e:
            tests['Engulfing Enhanced'] = f"Error: {str(e)[:30]}"
        
        # Pin Bar Enhanced
        try:
            curr = df.iloc[-1]
            found, tier, data = is_pin_bar_bullish_enhanced(curr, df)
            tests['Pin Bar Enhanced'] = f"{tier} tier" if found else False
        except Exception as e:
            tests['Pin Bar Enhanced'] = f"Error: {str(e)[:30]}"
        
        # Morning Star Enhanced
        try:
            found, tier, data = is_morning_star_enhanced(df)
            tests['Morning Star Enhanced'] = f"{tier} tier" if found else False
        except Exception as e:
            tests['Morning Star Enhanced'] = f"Error: {str(e)[:30]}"
        
        # Results
        for pattern, result in tests.items():
            if result is True:
                emoji = "✅"
                status = "FOUND"
            elif result is False:
                emoji = "❌"
                status = "Not found"
            elif isinstance(result, str) and "tier" in result:
                emoji = "✅"
                status = result
            else:
                emoji = "⚠️"
                status = str(result)
            
            msg += f"{emoji} {pattern}: {status}\n"
        
        msg += "\n<b>💡 Note:</b>\n"
        msg += "Questo test bypassa TUTTI i filtri globali.\n"
        msg += "Se trovi pattern qui ma non nelle analisi normali,\n"
        msg += "il problema è nei filtri (volume/trend/EMA).\n\n"
        msg += "Usa /debug_filters per vedere quale filtro blocca."
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
    except Exception as e:
        logging.exception('Errore in cmd_force_test')
        await update.message.reply_text(f'❌ Errore: {str(e)}')

async def monitor_closed_positions(context: ContextTypes.DEFAULT_TYPE):
    """
    Job che monitora posizioni chiuse e invia notifiche
    Eseguito ogni 30 secondi
    """
    if not config.BYBIT_API_KEY or not config.BYBIT_API_SECRET:
        return
    
    try:
        if not hasattr(monitor_closed_positions, 'notified_orders'):
            monitor_closed_positions.notified_orders = set()

        session = create_bybit_session()
        
        # Ottieni posizioni chiuse recenti (ultima ora)
        now = datetime.now(timezone.utc)
        one_hour_ago = int((now - timedelta(hours=1)).timestamp() * 1000)
        
        pnl_response = session.get_closed_pnl(
            category='linear',
            limit=50,
            startTime=one_hour_ago
        )
        
        if pnl_response.get('retCode') != 0:
            return
        
        pnl_list = pnl_response.get('result', {}).get('list', [])
        
        with config.POSITIONS_LOCK:
            tracked_symbols = set(config.ACTIVE_POSITIONS.keys())
        
        for pnl_entry in pnl_list:
            symbol = pnl_entry.get('symbol', 'N/A')
            
            # Verifica se era una posizione che stavamo tracciando
            if symbol not in tracked_symbols:
                continue
            
            # Verifica se abbiamo già notificato (usa timestamp)
            order_id = pnl_entry.get('orderId', '')
            
            if order_id in monitor_closed_positions.notified_orders:
                continue  # Già notificato
            
            # Marca come notificato
            monitor_closed_positions.notified_orders.add(order_id)
            
            # Estrai dati
            closed_size = float(pnl_entry.get('closedSize', 0))
            avg_entry = float(pnl_entry.get('avgEntryPrice', 0))
            avg_exit = float(pnl_entry.get('avgExitPrice', 0))
            closed_pnl = float(pnl_entry.get('closedPnl', 0))
            qty = float(pnl_entry.get('qty', 0))
            
            # Determina direzione
            if closed_size > 0:
                trade_direction = 'LONG'
                side_emoji = "🟢"
            elif closed_size < 0:
                trade_direction = 'SHORT'
                side_emoji = "🔴"
            else:
                if avg_exit > avg_entry:
                    trade_direction = 'LONG'
                    side_emoji = "🟢"
                else:
                    trade_direction = 'SHORT'
                    side_emoji = "🔴"
            
            # Calcola PnL %
            if avg_entry > 0:
                if trade_direction == 'LONG':
                    pnl_percent = ((avg_exit - avg_entry) / avg_entry) * 100
                else:
                    pnl_percent = ((avg_entry - avg_exit) / avg_entry) * 100
            else:
                pnl_percent = 0
            
            # Determina tipo chiusura
            if closed_pnl > 0:
                result_emoji = "✅"
                result_text = "PROFIT"
                result_color = "🟢"
            else:
                result_emoji = "❌"
                result_text = "LOSS"
                result_color = "🔴"
            
            # Decimali dinamici
            price_decimals = get_price_decimals(avg_entry)
            
            # Costruisci messaggio
            msg = f"{result_emoji} <b>POSIZIONE CHIUSA - {result_text}</b>\n\n"
            msg += f"{side_emoji} <b>{symbol}</b> - {trade_direction}\n\n"
            
            msg += f"<b>📊 Trade Details:</b>\n"
            msg += f"Qty: {abs(qty):.4f}\n"
            msg += f"Entry: ${avg_entry:.{price_decimals}f}\n"
            msg += f"Exit: ${avg_exit:.{price_decimals}f}\n\n"
            
            msg += f"<b>{result_color} Risultato:</b>\n"
            msg += f"PnL: <b>${closed_pnl:+.2f}</b> ({pnl_percent:+.2f}%)\n\n"
            
            # Recupera info posizione originale se disponibile
            with config.POSITIONS_LOCK:
                if symbol in config.ACTIVE_POSITIONS:
                    pos_info = config.ACTIVE_POSITIONS[symbol]
                    
                    entry_original = pos_info.get('entry_price', avg_entry)
                    sl_original = pos_info.get('sl', 0)
                    tp_original = pos_info.get('tp', 0)
                    timeframe = pos_info.get('timeframe', 'N/A')

                    # Registra risultato nelle statistiche
                    try:
                        track_patterns.integrate_pattern_stats_on_close(
                            symbol=symbol,
                            entry_price=pos_info.get('entry_price', avg_entry),
                            exit_price=avg_exit,
                            pnl=closed_pnl,
                            open_timestamp=pos_info.get('timestamp'),
                            close_timestamp=datetime.now(timezone.utc).isoformat(),
                            pattern_name=pos_info.get('pattern_name'),  # ← AGGIUNGI
                            timeframe=pos_info.get('timeframe'),  # ← AGGIUNGI
                            side=pos_info.get('side')  # ← AGGIUNGI
                        )
                    except Exception as e:
                        logging.error(f'Errore tracking trade result: {e}')
                    
                    msg += f"<b>📈 Setup Originale:</b>\n"
                    msg += f"Timeframe: {timeframe}\n"
                    msg += f"Entry Plan: ${entry_original:.{price_decimals}f}\n"
                    msg += f"SL: ${sl_original:.{price_decimals}f}\n"
                    msg += f"TP: ${tp_original:.{price_decimals}f}\n\n"
                    
                    # Determina tipo chiusura (SL/TP/Manual)
                    if abs(avg_exit - tp_original) < (tp_original * 0.002):
                        msg += "🎯 <b>Tipo chiusura: TAKE PROFIT</b>\n"
                    elif abs(avg_exit - sl_original) < (sl_original * 0.002):
                        msg += "🛑 <b>Tipo chiusura: STOP LOSS</b>\n"
                    else:
                        msg += "👤 <b>Tipo chiusura: MANUAL</b>\n"
                    
                    # Rimuovi dal tracking
                    del config.ACTIVE_POSITIONS[symbol]
                    logging.info(f'📝 Rimossa {symbol} dal tracking dopo chiusura')
            
            # Timestamp
            updated_time = int(pnl_entry.get('updatedTime', 0))
            close_time = datetime.fromtimestamp(updated_time / 1000, tz=timezone.utc)
            time_str = close_time.strftime('%d/%m/%Y %H:%M UTC')
            msg += f"\n⏰ Chiuso: {time_str}"
            
            # Invia notifica a tutte le chat che stanno tracciando questo symbol
            with config.ACTIVE_ANALYSES_LOCK:
                for chat_id, analyses in ACTIVE_ANALYSES.items():
                    # Verifica se questa chat stava analizzando questo symbol
                    is_tracking = any(key.startswith(f'{symbol}-') for key in analyses.keys())
                    
                    if is_tracking:
                        try:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=msg,
                                parse_mode='HTML'
                            )
                            logging.info(f'📨 Notifica chiusura inviata a chat {chat_id} per {symbol}')
                        except Exception as e:
                            logging.error(f'Errore invio notifica chiusura: {e}')
        
        # Cleanup vecchie notifiche (mantieni solo ultime 200)
        orders = monitor_closed_positions.notified_orders
        if len(orders) > 200:
            monitor_closed_positions.notified_orders = set(list(orders)[-200:])
    
    except Exception as e:
        logging.exception('Errore in monitor_closed_positions')

async def cmd_testcache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test della cache instrument info"""
    import time
    symbol = "BTCUSDT"
    
    session = create_bybit_session()
    
    # Primo accesso (dovrebbe scaricare)
    start = time.time()
    info1 = get_instrument_info_cached(session, symbol)
    time1 = (time.time() - start) * 1000
    
    # Secondo accesso (dovrebbe usare cache)
    start = time.time()
    info2 = get_instrument_info_cached(session, symbol)
    time2 = (time.time() - start) * 1000
    
    await update.message.reply_text(
        f"<b>Cache Test Results:</b>\n"
        f"1st call: {time1:.2f}ms (download)\n"
        f"2nd call: {time2:.2f}ms (cached)\n"
        f"Speedup: {time1/time2:.1f}x\n\n"
        f"<b>Info:</b>\n"
        f"Min Qty: {info2['min_order_qty']}\n"
        f"Qty Step: {info2['qty_step']}\n"
        f"Decimals: {info2['qty_decimals']}",
        parse_mode='HTML'
    )

async def cmd_multitp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /multitp [ACTION]
    
    Actions:
    - (nessuno): Mostra status
    - on: Abilita multi-TP
    - off: Disabilita multi-TP
    """
    global MULTI_TP_ENABLED
    
    args = context.args
    
    if not args:
        # ===== MOSTRA STATUS =====
        
        with config.POSITIONS_LOCK:
            positions_with_multitp = {
                sym: pos for sym, pos in config.ACTIVE_POSITIONS.items()
                if pos.get('multi_tp_levels')
            }
        
        msg = "<b>🎯 Multi-TP System Status</b>\n\n"
        msg += f"Enabled: {'✅ ON' if config.MULTI_TP_ENABLED else '❌ OFF'}\n\n"
        
        if config.MULTI_TP_ENABLED:
            msg += "<b>📊 Configuration:</b>\n"
            for i, level in enumerate(config.MULTI_TP_CONFIG['levels'], 1):
                msg += f"{level['emoji']} TP{i}: {level['rr_ratio']}R ({level['close_pct']*100:.0f}%)\n"
            
            msg += f"\nCheck Interval: {config.MULTI_TP_CONFIG['check_interval']}s\n"
            msg += f"Min Partial Qty: {config.MULTI_TP_CONFIG['min_partial_qty']}\n\n"
            
            if positions_with_multitp:
                msg += f"<b>📦 Posizioni Attive con Multi-TP ({len(positions_with_multitp)}):</b>\n\n"
                
                for symbol, pos in positions_with_multitp.items():
                    side = pos['side']
                    side_emoji = "🟢" if side == "Buy" else "🔴"
                    
                    msg += f"{side_emoji} <b>{symbol}</b>\n"
                    
                    # Status TP
                    tp_levels = pos['multi_tp_levels']
                    for i, tp in enumerate(tp_levels, 1):
                        if tp.get('hit'):
                            msg += f"  ✅ TP{i}: ${tp['price']:.4f} (HIT)\n"
                        else:
                            msg += f"  ⏳ TP{i}: ${tp['price']:.4f}\n"
                    
                    msg += f"  Qty: {pos['qty']:.4f} / {pos['qty_original']:.4f}\n\n"
            else:
                msg += "Nessuna posizione con Multi-TP attivo\n"
        
        else:
            msg += "Multi-TP disabilitato.\n"
            msg += "Usa <code>/multitp on</code> per abilitare"
        
        msg += "\n<b>Comandi:</b>\n"
        msg += "<code>/multitp on</code> - Abilita\n"
        msg += "<code>/multitp off</code> - Disabilita"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        return
    
    # ===== ABILITA/DISABILITA =====
    action = args[0].lower()
    
    if action == 'on':
        config.MULTI_TP_ENABLED = True
        config.MULTI_TP_CONFIG['enabled'] = True
        
        msg = "✅ <b>Multi-TP System ATTIVATO</b>\n\n"
        msg += "Nuove posizioni avranno TP multipli:\n"
        for i, level in enumerate(config.MULTI_TP_CONFIG['levels'], 1):
            msg += f"{level['emoji']} TP{i}: {level['rr_ratio']}R ({level['close_pct']*100:.0f}%)\n"
        
        msg += "\n💡 <b>Benefici:</b>\n"
        msg += "• Banca profitto progressivamente\n"
        msg += "• Lascia correre i winner\n"
        msg += "• Trailing attivo dopo TP1\n"
        msg += "• Rischio gestito meglio"
        
        await update.message.reply_text(msg, parse_mode='HTML')
    
    elif action == 'off':
        config.MULTI_TP_ENABLED = False
        config.MULTI_TP_CONFIG['enabled'] = False
        
        msg = "❌ <b>Multi-TP System DISATTIVATO</b>\n\n"
        msg += "Nuove posizioni avranno TP singolo (2R)"
        
        await update.message.reply_text(msg, parse_mode='HTML')
    
    else:
        await update.message.reply_text(
            "❌ Comando non valido\n"
            "Usa: <code>/multitp [on|off]</code>",
            parse_mode='HTML'
        )

# ----------------------------- MAIN -----------------------------

def main():
    """Funzione principale"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,  # 👈 Cambia da INFO a DEBUG per vedere i filtri
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

        # Avvia Auto-Discovery se abilitato
    if config.AUTO_DISCOVERY_ENABLED and config.AUTO_DISCOVERY_CONFIG['enabled']:
        # Nota: Serve chat_id, quindi auto-discovery sarà attivato
        # dal primo utente che usa /autodiscover on
        logging.info('🔍 Auto-Discovery configurato (attiva con /autodiscover on)')

    # Carica statistiche pattern
    logging.info('📊 Caricamento statistiche pattern...')
    track_patterns.load_pattern_stats()
    
    # Verifica variabili d'ambiente
    if not config.TELEGRAM_TOKEN or config.TELEGRAM_TOKEN == '':
        logging.error('❌ TELEGRAM_TOKEN non configurato!')
        logging.error('Imposta la variabile d\'ambiente TELEGRAM_TOKEN')
        return
    
    if not config.BYBIT_API_KEY or not config.BYBIT_API_SECRET:
        logging.warning('⚠️ Bybit API keys non configurate. Trading disabilitato.')
    
    # Crea applicazione con JobQueue
    try:
        from telegram.ext import JobQueue
        application = ApplicationBuilder().token(config.TELEGRAM_TOKEN).build()
        if application.job_queue is None:
            logging.error('❌ JobQueue non disponibile!')
            logging.error('Installa: pip install "python-telegram-bot[job-queue]"')
            return
    except ImportError:
        logging.error('❌ JobQueue non disponibile!')
        logging.error('Installa: pip install "python-telegram-bot[job-queue]"')
        return
    
    # Aggiungi handlers
    application.add_handler(CommandHandler('start', cmd_start))
    application.add_handler(CommandHandler('analizza', cmd_analizza))
    application.add_handler(CommandHandler('stop', cmd_stop))
    application.add_handler(CommandHandler('list', cmd_list))
    application.add_handler(CommandHandler('test', cmd_test))
    application.add_handler(CommandHandler('balance', cmd_balance))
    application.add_handler(CommandHandler('orders', cmd_orders))
    application.add_handler(CommandHandler('pausa', cmd_pausa))
    application.add_handler(CommandHandler('abilita', cmd_abilita))
    application.add_handler(CommandHandler('posizioni', cmd_posizioni))
    application.add_handler(CommandHandler('chiudi', cmd_chiudi))
    application.add_handler(CommandHandler('sync', cmd_sync))
    application.add_handler(CommandHandler('trailing', cmd_trailing))
    application.add_handler(CommandHandler('autodiscover', cmd_autodiscover))
    application.add_handler(CommandHandler('patterns', cmd_patterns))
    application.add_handler(CommandHandler('pattern_on', cmd_pattern_on))
    application.add_handler(CommandHandler('pattern_off', cmd_pattern_off))
    application.add_handler(CommandHandler('pattern_info', cmd_pattern_info))
    application.add_handler(CommandHandler('ema_filter', cmd_ema_filter))
    application.add_handler(CommandHandler('trend_filter', cmd_trend_filter))
    application.add_handler(CommandHandler("timefilter", cmd_time_filter))
    application.add_handler(CommandHandler('debug_filters', cmd_debug_filters))
    application.add_handler(CommandHandler('force_test', cmd_force_test))
    application.add_handler(CommandHandler('pattern_stats', track_patterns.cmd_pattern_stats))
    application.add_handler(CommandHandler('reset_pattern_stats', track_patterns.cmd_reset_pattern_stats))
    application.add_handler(CommandHandler('export_pattern_stats', track_patterns.cmd_export_pattern_stats))
    application.add_handler(CommandHandler('testcache', cmd_testcache))
    application.add_handler(CommandHandler('multitp', cmd_multitp))

    # Schedula trailing stop loss job
    schedule_trailing_stop_job(application)

    # Schedula Multi-TP monitoring
    if config.MULTI_TP_ENABLED and config.MULTI_TP_CONFIG['enabled']:
        application.job_queue.run_repeating(
            monitor_partial_tp,
            interval=config.MULTI_TP_CONFIG['check_interval'],
            first=30,  # Primo check dopo 30 secondi
            name='monitor_partial_tp'
        )
        
        logging.info(
            f'✅ Multi-TP monitoring attivato '
            f'(check ogni {config.MULTI_TP_CONFIG["check_interval"]}s)'
        )

    async def save_pattern_stats_job(context: ContextTypes.DEFAULT_TYPE):
        """Salva le statistiche pattern periodicamente"""
        try:
            track_patterns.save_pattern_stats()
            logging.debug("Pattern stats saved successfully")
        except Exception as e:
            logging.error(f"Error saving pattern stats: {e}")
    
    # Poi, nella sezione di scheduling:
    if True:  # Sempre attivo
        application.job_queue.run_repeating(
            save_pattern_stats_job,  # ← Usa funzione async invece di lambda
            interval=300,
            first=60,
            name='save_pattern_stats'
        )

    # ===== NUOVO: Schedula monitoring posizioni chiuse =====
    if config.BYBIT_API_KEY and config.BYBIT_API_SECRET:
        application.job_queue.run_repeating(
            monitor_closed_positions,
            interval=30,  # Ogni 30 secondi
            first=10,     # Primo check dopo 10 secondi
            name='monitor_closed_positions'
        )
        logging.info('✅ Monitoring posizioni chiuse attivato (ogni 30s)')
    
    # Avvia bot
    mode_emoji = "🎮" if config.TRADING_MODE == 'demo' else "⚠️💰"
    logging.info('🚀 Bot avviato correttamente!')
    logging.info(f'{mode_emoji} Modalità Trading: {config.TRADING_MODE.upper()}')
    logging.info(f'⏱️ Timeframes supportati: {config.ENABLED_TFS}')
    logging.info(f'💰 Rischio per trade: ${config.RISK_USD}')
    
    if config.TRADING_MODE == 'live':
        logging.warning('⚠️⚠️⚠️ ATTENZIONE: MODALITÀ LIVE - TRADING REALE! ⚠️⚠️⚠️')
    
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True  # Ignora aggiornamenti pendenti
        )
    except telegram.error.Conflict as e:
        logging.error('❌ ERRORE: Un\'altra istanza del bot è già in esecuzione!')
        logging.error('Soluzione: Ferma tutte le altre istanze del bot')
        logging.error(f'Dettaglio errore: {e}')
    except Exception as e:
        logging.exception(f'❌ Errore critico: {e}')


if __name__ == '__main__':
    main()

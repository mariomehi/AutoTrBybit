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
from datetime import datetime, timezone
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

# ----------------------------- CONFIG -----------------------------
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
BYBIT_API_KEY = os.environ.get('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.environ.get('BYBIT_API_SECRET', '')

# Scegli l'ambiente di trading
# 'demo' = Demo Trading (fondi virtuali)
# 'live' = Trading Reale (ATTENZIONE: soldi veri!)
TRADING_MODE = os.environ.get('TRADING_MODE', 'demo')

# Strategy parameters
VOLUME_FILTER = True
ATR_MULT_SL = 1.5
ATR_MULT_TP = 2.0
RISK_USD = 10.0
ENABLED_TFS = ['1m','3m','5m','15m','30m','1h','4h']

# Flag globale: abilita/disabilita volume filter
VOLUME_FILTER_ENABLED = True  # Default: abilitato
# Modalità volume filter
VOLUME_FILTER_MODE = 'pattern-only'  # 'strict', 'adaptive', 'pattern-only'
# Threshold per diversi modi
VOLUME_THRESHOLDS = {
    'strict': 2.0,      # Volume > 2x media (originale)
    'adaptive': 1.3,    # Volume > 1.3x media (rilassato)
    'pattern-only': 0   # No check globale, solo nei pattern
}

# EMA-based Stop Loss System
USE_EMA_STOP_LOSS = True  # Usa EMA invece di ATR per stop loss
EMA_STOP_LOSS_CONFIG = {
    # Per ogni timeframe, quale EMA usare come stop loss dinamico
    '5m': 'ema10',   # Scalping: EMA 10
    '15m': 'ema10',  # Scalping: EMA 10
    '30m': 'ema10',  # Day: EMA 10
    '1h': 'ema60',   # Day: EMA 60 (più spazio)
    '4h': 'ema60',   # Swing: EMA 60 (più conservativo)
}

# Trailing Stop Loss System
TRAILING_STOP_ENABLED = True  # Abilita/disabilita trailing SL
TRAILING_CONFIG = {
    'activation_percent': 0.5,  # Attiva trailing dopo +0.5% profit
    'ema_buffer': 0.002,  # Buffer 0.2% sotto EMA 10
    'never_back': True,  # SL non torna mai indietro
    'check_interval': 150,  # Check ogni 5 minuti (300 secondi)
}

# Timeframe di riferimento per EMA 10 trailing
# Per ogni TF entry, usa questo TF per calcolare EMA 10
TRAILING_EMA_TIMEFRAME = {
    '1m': '5m',   # Entry su 1m → EMA 10 da 5m
    '3m': '5m',   # Entry su 3m → EMA 10 da 5m
    '5m': '5m',  # Entry su 5m → EMA 10 da 15m
    '15m': '30m', # Entry su 15m → EMA 10 da 30m
    '30m': '1h',  # Entry su 30m → EMA 10 da 1h
    '1h': '4h',   # Entry su 1h → EMA 10 da 4h
    '4h': '4h',   # Entry su 4h → EMA 10 da 4h stesso
}

# Buffer EMA Stop Loss (% sotto l'EMA per evitare falsi breakout)
EMA_SL_BUFFER = 0.002  # 0.2% sotto l'EMA
# Esempio: se EMA 10 = $100, SL = $100 * (1 - 0.002) = $99.80

# Symbol-specific risk overrides (opzionale)
SYMBOL_RISK_OVERRIDE = {
    # Esempio: per MONUSDT usa solo $5 invece di $10
    # 'MONUSDT': 5.0,
}

# EMA Filter System
EMA_FILTER_ENABLED = True  # Abilita/disabilita filtro EMA

# Modalità EMA Filter
# 'strict' = Pattern valido SOLO se tutte le condizioni EMA sono rispettate
# 'loose' = Pattern valido ma segnala se condizioni EMA non ottimali
# 'off' = Nessun filtro EMA (solo pattern)
EMA_FILTER_MODE = 'loose'  # 'strict', 'loose', 'off'

# Configurazione EMA per diversi timeframe
EMA_CONFIG = {
    # Scalping (5m, 15m) - Focus su EMA veloci
    'scalping': {
        'timeframes': ['5m', '15m'],
        'rules': {
            # MUST 1: Prezzo sopra EMA 10
            'price_above_ema10': True,
            # MUST 2: Prezzo sopra EMA 60 (trend filter) 👈 NUOVO
            'price_above_ema60': True,
            # BONUS: EMA 5 sopra EMA 10 (momentum forte)
            'ema5_above_ema10': True,
            # GOLD: Pattern vicino a EMA 10 (pullback)
            'near_ema10': False  # Opzionale
        }
    },
    
    # Day Trading (30m, 1h) - Balance tra veloce e medio
    'daytrading': {
        'timeframes': ['30m', '1h'],
        'rules': {
            # MUST: Prezzo sopra EMA 60
            'price_above_ema60': True,
            # BONUS: EMA 10 sopra EMA 60
            'ema10_above_ema60': True,
            # GOLD: Pattern vicino a EMA 60 (pullback zone)
            'near_ema60': True
        }
    },
    
    # Swing Trading (4h, 1d) - Focus su trend lungo
    'swing': {
        'timeframes': ['4h'],
        'rules': {
            # MUST: Prezzo sopra EMA 223 (trend principale)
            'price_above_ema223': True,
            # BONUS: EMA 60 sopra EMA 223
            'ema60_above_ema223': True,
            # GOLD: Pattern vicino a EMA 223 (bounce)
            'near_ema223': True
        }
    }
}

# Auto-Discovery System
AUTO_DISCOVERY_ENABLED = True  # Abilita/disabilita auto-discovery
AUTO_DISCOVERY_CONFIG = {
    'enabled': True,
    'top_count': 10,  # Top 10 symbols
    'timeframe': '5m',  # Timeframe da analizzare
    'autotrade': True,  # Autotrade per auto-discovery (False = solo notifiche)
    'update_interval': 14400,  # 12 ore in secondi (12 * 60 * 60)
    'min_volume_usdt': 5000000,  # Min volume 24h: 10M USDT
    'min_price_change': 5.0,  # Min variazione 24h: +5%
    'max_price_change': 80.0,  # Max variazione 24h: +50% (evita pump & dump)
    'exclude_symbols': ['USDCUSDT', 'TUSDUSDT', 'BUSDUSDT'],  # Stablecoins da escludere
    'sorting': 'price_change_percent',  # 'price_change_percent' o 'volume'
}

# Storage per simboli auto-discovered
AUTO_DISCOVERED_SYMBOLS = set()
AUTO_DISCOVERED_LOCK = threading.Lock()

# Pattern Management System
AVAILABLE_PATTERNS = {
        'volume_spike_breakout': {
        'name': 'Volume Spike Breakout',
        'enabled': True,  # 👈 ABILITATO di default
        'description': 'Breakout con volume 3x+ e momentum forte (Best per crypto)',
        'side': 'Buy',
        'emoji': '📊💥'
    },
        'breakout_retest': {  # 👈 NUOVO
        'name': 'Breakout + Retest',
        'enabled': True,  # Abilita di default
        'description': 'Breakout → Pullback → Retest con bounce (Win rate 60-70%)',
        'side': 'Buy',
        'emoji': '🔄📈'
    },
        'triple_touch_breakout': {  # 👈 NUOVO
        'name': 'Triple Touch Breakout',
        'enabled': True,
        'description': '3 tocchi resistance (2 rejection + breakout) sopra EMA 60 (62-72% win)',
        'side': 'Buy',
        'emoji': '🎯3️⃣'
    },
        'liquidity_sweep_reversal': {  # 👈 NUOVO (alta priorità!)
        'name': 'Liquidity Sweep + Reversal',
        'enabled': True,
        'description': 'Smart Money: sweep low + reversal (istituzionale)',
        'side': 'Buy',
        'emoji': '💎'
    },
        'sr_bounce': {
        'name': 'Support/Resistance Bounce',
        'enabled': True,
        'description': 'Rimbalzo su livello S/R con rejection',
        'side': 'Buy',
        'emoji': '🎯'
    },
    'bullish_comeback': {
        'name': 'Bullish Comeback',
        'enabled': True,
        'description': 'Inversione/rigetto rialzista (2 varianti)',
        'side': 'Buy',
        'emoji': '🔄'
    },
    'compression_breakout': { 
        'name': 'Compression Breakout (Enhanced)',
        'enabled': True,
        'description': 'Breakout con volume 1.8x+, RSI 50-70, no HTF resistance',
        'side': 'Buy',
        'emoji': '💥'
    },
    'bullish_flag_breakout': {
        'name': 'Bullish Flag Breakout (Enhanced)',
        'enabled': True,
        'description': 'Breakout volume 2x+, flag 3-8 candele, pole >0.8%',
        'side': 'Buy',
        'emoji': '🚩'
    },
    'morning_star_ema_breakout': {  # 👈 NUOVO
        'name': 'Morning Star + EMA Breakout',
        'enabled': True,
        'description': 'Morning Star con rottura EMA 5,10 al rialzo',
        'side': 'Buy',
        'emoji': '⭐💥'
    },
    'bullish_engulfing': {
        'name': 'Bullish Engulfing',
        'enabled': True,
        'description': 'Candela rialzista ingloba ribassista',
        'side': 'Buy',
        'emoji': '🟢'
    },
    'hammer': {
        'name': 'Hammer',
        'enabled': True,
        'description': 'Corpo piccolo in alto, ombra lunga sotto',
        'side': 'Buy',
        'emoji': '🔨'
    },
    'pin_bar_bullish': {
        'name': 'Pin Bar Bullish',
        'enabled': True,
        'description': 'Ombra inferiore molto lunga',
        'side': 'Buy',
        'emoji': '📍'
    },
    'morning_star': {
        'name': 'Morning Star',
        'enabled': False,
        'description': '3 candele: ribassista, piccola, rialzista',
        'side': 'Buy',
        'emoji': '⭐'
    },
    'three_white_soldiers': {
        'name': 'Three White Soldiers',
        'enabled': False,
        'description': '3 candele rialziste consecutive forti',
        'side': 'Buy',
        'emoji': '⬆️'
    },
    # Pattern SELL (disabilitati di default)
    'bearish_engulfing': {
        'name': 'Bearish Engulfing',
        'enabled': False,
        'description': 'Candela ribassista ingloba rialzista',
        'side': 'Sell',
        'emoji': '🔴'
    },
    'shooting_star': {
        'name': 'Shooting Star',
        'enabled': False,
        'description': 'Corpo piccolo in basso, ombra lunga sopra',
        'side': 'Sell',
        'emoji': '💫'
    },
    'pin_bar_bearish': {
        'name': 'Pin Bar Bearish',
        'enabled': False,
        'description': 'Ombra superiore molto lunga',
        'side': 'Sell',
        'emoji': '📍'
    },
    'evening_star': {
        'name': 'Evening Star',
        'enabled': False,
        'description': '3 candele: rialzista, piccola, ribassista',
        'side': 'Sell',
        'emoji': '🌙'
    },
    'three_black_crows': {
        'name': 'Three Black Crows',
        'enabled': False,
        'description': '3 candele ribassiste consecutive forti',
        'side': 'Sell',
        'emoji': '⬇️'
    },
    'doji': {
        'name': 'Doji',
        'enabled': False,
        'description': 'Indecisione, corpo molto piccolo',
        'side': 'Both',
        'emoji': '➖'
    }
}

# Lock per modifiche thread-safe
PATTERNS_LOCK = threading.Lock()

# Klines map
BYBIT_INTERVAL_MAP = {
    '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', 
    '1h': '60', '4h': '240', '1d': 'D'
}

# Interval to seconds mapping
INTERVAL_SECONDS = {
    '1m': 60, '3m': 180, '5m': 300, '15m': 900, 
    '30m': 1800, '1h': 3600, '4h': 14400, '1d': 86400
}

# Active analyses storage
ACTIVE_ANALYSES = {}
ACTIVE_ANALYSES_LOCK = threading.Lock()

# Full notifications (include grafici senza pattern): chat_id -> set of "SYMBOL-TIMEFRAME" keys
# DEFAULT: solo pattern. Se symbol-tf è in questo set, invia TUTTE le notifiche
FULL_NOTIFICATIONS = {}
FULL_NOTIFICATIONS_LOCK = threading.Lock()

# Active positions tracking: symbol -> order_info
# order_info contiene:
# - side: 'Buy' o 'Sell'
# - qty: quantità
# - entry_price: prezzo di entrata
# - sl: stop loss corrente
# - tp: take profit
# - order_id: ID ordine Bybit
# - timestamp: quando è stato aperto
# - timeframe: TF su cui è stato rilevato
# - trailing_active: se trailing è attivo
# - highest_price: prezzo massimo raggiunto (per trailing)
ACTIVE_POSITIONS = {}
POSITIONS_LOCK = threading.Lock()

# Bybit endpoints
BYBIT_ENDPOINTS = {
    'demo': 'https://api-demo.bybit.com',  # Demo trading
    'live': 'https://api.bybit.com'        # Trading reale
}
BYBIT_PUBLIC_REST = 'https://api.bybit.com'  # Dati di mercato sempre da mainnet


def create_bybit_session():
    """Crea sessione Bybit per trading (Demo o Live)"""
    if BybitHTTP is None:
        raise RuntimeError('pybit non disponibile. Installa: pip install pybit>=5.0')
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError('BYBIT_API_KEY e BYBIT_API_SECRET devono essere configurate')
    
    # Determina l'endpoint in base alla modalità
    base_url = BYBIT_ENDPOINTS.get(TRADING_MODE, BYBIT_ENDPOINTS['demo'])
    
    logging.info(f'🔌 Connessione Bybit - Modalità: {TRADING_MODE.upper()}')
    logging.info(f'📡 Endpoint: {base_url}')
    
    session = BybitHTTP(
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_API_SECRET,
        testnet=False,  # Non usiamo testnet
        demo=True if TRADING_MODE == 'demo' else False  # Usa demo se configurato
    )
    
    return session

# ----------------------------- UTILITIES -----------------------------

def bybit_get_klines(symbol: str, interval: str, limit: int = 200):
    """
    Ottiene klines da Bybit v5 public API
    Returns: DataFrame con OHLCV
    """
    itv = BYBIT_INTERVAL_MAP.get(interval)
    if itv is None:
        raise ValueError(f'Timeframe non supportato: {interval}')

    url = f'{BYBIT_PUBLIC_REST}/v5/market/kline'
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


def volume_confirmation(
    df: pd.DataFrame, 
    min_ratio: float = 1.5,
    symbol: str = None,
    is_auto_discovered: bool = False
    ) -> bool:
    """
    Volume Confirmation - ADAPTIVE VERSION
    
    LOGICA:
    1. Se symbol è auto-discovered → CHECK RILASSATO (o skip)
       (già filtrato per volume 24h alto)
    
    2. Se symbol è manuale → CHECK STRETTO
       (potrebbe essere low volume)
    
    3. Se mode = 'pattern-only' → SKIP (ogni pattern decide)
    
    Args:
        df: DataFrame OHLCV
        min_ratio: Ratio minimo (default 1.5x)
        symbol: Symbol name (per check auto-discovery)
        is_auto_discovered: Flag esplicito
    
    Returns:
        bool: True se volume OK
    """
    # === MODE 1: PATTERN-ONLY (no global check) ===
    if VOLUME_FILTER_MODE == 'pattern-only':
        logging.debug(f'Volume check: SKIPPED (pattern-only mode)')
        return True
    
    # === MODE 2: ADAPTIVE (rilassato per auto-discovered) ===
    if VOLUME_FILTER_MODE == 'adaptive':
        # Check se symbol è auto-discovered
        with AUTO_DISCOVERED_LOCK:
            is_auto = symbol in AUTO_DISCOVERED_SYMBOLS if symbol else False
        
        if is_auto or is_auto_discovered:
            # Symbol ad alto volume 24h → check rilassato
            min_ratio = 1.2  # Abbassa threshold
            logging.debug(f'{symbol}: Auto-discovered, threshold rilassato (1.2x)')
    
    # === MODE 3: STRICT (check normale) ===
    # Usa min_ratio passato come parametro
    
    # === VOLUME CALCULATION ===
    if 'volume' not in df.columns:
        logging.error('❌ Volume column NOT FOUND')
        return False
    
    if len(df) < 20:
        logging.warning(f'⚠️ Insufficient data: {len(df)} rows')
        return False
    
    vol = df['volume']
    
    # Check NaN
    if vol.isna().all():
        logging.error('❌ All volume values are NaN')
        return False
    
    # Calcola media (esclude corrente)
    avg_vol = vol.iloc[-20:-1].mean()
    current_vol = vol.iloc[-1]
    
    # Validation
    if pd.isna(avg_vol) or pd.isna(current_vol):
        logging.error(f'❌ Volume NaN: avg={avg_vol}, current={current_vol}')
        return False
    
    # === FALLBACK: Se avg_vol = 0 MA symbol è auto-discovered ===
    if avg_vol == 0:
        with AUTO_DISCOVERED_LOCK:
            is_auto = symbol in AUTO_DISCOVERED_SYMBOLS if symbol else False
        
        if is_auto or is_auto_discovered:
            # Symbol top gainer → probabilmente dati incompleti, PERMETTI
            logging.warning(
                f'⚠️ {symbol}: avg_vol=0 MA è auto-discovered → ALLOW'
            )
            return True
        else:
            # Symbol manuale con avg_vol=0 → BLOCCA
            logging.error(f'❌ {symbol}: avg_vol=0 → BLOCK')
            return False
    
    # Calcola ratio
    ratio = current_vol / avg_vol
    
    result = ratio > min_ratio
    
    if result:
        logging.debug(
            f'✅ Volume OK: {ratio:.2f}x (threshold {min_ratio}x)'
        )
    else:
        logging.info(
            f'⚠️ Volume insufficient: {ratio:.2f}x (need {min_ratio}x+)'
        )
    
    return result


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


def is_uptrend_structure(df: pd.DataFrame, lookback: int = 10) -> bool:
    """
    Filtro Market Structure - Higher Highs + Higher Lows
    Solo trade in direzione del trend
    
    Args:
        df: DataFrame OHLCV
        lookback: Candele da analizzare
    
    Returns:
        True se uptrend confermato
    """
    if len(df) < lookback + 3:
        return False
    
    highs = df['high'].iloc[-lookback:]
    lows = df['low'].iloc[-lookback:]
    
    # Dividi in due metà
    split = lookback // 2
    
    recent_high = highs.iloc[-split:].max()
    previous_high = highs.iloc[:-split].max()
    
    recent_low = lows.iloc[-split:].min()
    previous_low = lows.iloc[:-split].min()
    
    # Higher High AND Higher Low = uptrend
    return recent_high > previous_high and recent_low > previous_low
    

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
    if not EMA_FILTER_ENABLED or EMA_FILTER_MODE == 'off':
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
    config = None
    for strategy, cfg in EMA_CONFIG.items():
        if timeframe in cfg['timeframes']:
            config = cfg
            break
    
    if not config:
        # Default: usa day trading config
        config = EMA_CONFIG['daytrading']
    
    rules = config['rules']
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
    if EMA_FILTER_MODE == 'strict':
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
    
    BEST PATTERN per Crypto Intraday
    Win Rate: 58-62% (5m), 62-68% (15m)
    
    CONDIZIONI:
    1. Volume corrente > 3x media (spike massive)
    2. Candela verde con corpo forte (>60% range)
    3. Close vicino al high (<15% ombra superiore)
    4. Breakout EMA 10 al rialzo
    5. Prezzo sopra EMA 60 (trend filter)
    6. EMA 10 > EMA 60 (alignment)
    7. Momentum non già esteso
    
    Returns:
        (found: bool, pattern_data: dict or None)
    """
    if len(df) < 60:
        return (False, None)
    
    # Candele
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # === CHECK 1: VOLUME SPIKE (3x minimo) ===
    vol = df['volume']
    
    if len(vol) < 20:
        return (False, None)
    
    avg_vol = vol.iloc[-20:-1].mean()
    current_vol = vol.iloc[-1]
    
    if avg_vol == 0:
        return (False, None)
    
    volume_ratio = current_vol / avg_vol
    
    if volume_ratio < 3.0:
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
    🥉 SUPPORT/RESISTANCE BOUNCE
    
    Win Rate: 52-58% (5m), 56-62% (15m)
    Risk:Reward: 1.6:1 medio
    
    COME FUNZIONA:
    ============================================
    1. Identifica livelli S/R significativi (ultimi 50 periodi)
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
       - Volume > 1.5x media (check_patterns)
       - Uptrend structure (check_patterns)
       - ATR expanding (warning only)
    
    ✅ Filtri INTERNI:
       - Volume > 1.2x media (meno stretto di Sweep)
       - Support toccato 3+ volte (valido)
       - Rejection: wick >= corpo
       - Close sopra/vicino EMA 10 (trend OK)
       - Distanza < 2% da EMA 60 (non troppo lontano)
    
    EMA USATE:
    ============================================
    - EMA 10: Check trend breve (momentum)
    - EMA 60: Check trend medio (distanza max 2%)
    - NO EMA 5, 223 (non rilevanti per questo pattern)
    
    Returns:
        (found: bool, pattern_data: dict or None)
    """
    if len(df) < 50:
        return (False, None)
    
    curr = df.iloc[-1]
    
    # === STEP 1: IDENTIFICA SUPPORT LEVEL ===
    # Support = zona con più "touches" negli ultimi 50 periodi
    lookback_lows = df['low'].iloc[-50:-1]
    
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
    
    # Volume minimo 1.2x (meno stretto di Sweep che richiede 2x)
    # Perché: bounce su S/R può avere volume normale
    if vol_ratio < 1.2:
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
    
    # Distanza massima 2% da EMA 60
    distance_to_ema60 = abs(curr['close'] - curr_ema60) / curr_ema60
    
    if distance_to_ema60 > 0.02:
        return (False, None)
    
    # === STEP 7: QUALITY BONUS (opzionale) ===
    # Più vicino a EMA 60 = qualità migliore
    near_ema60 = distance_to_ema60 < 0.01  # Entro 1%
    
    # Rejection strength (quanto è forte il wick rispetto al corpo)
    rejection_strength = lower_wick / body if body > 0 else 1.0
    
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
        'tier': 1  # High priority (Tier 1)
    }
    
    return (True, pattern_data)

def is_bullish_engulfing(prev, curr):
    """
    Pattern: Bullish Engulfing
    Candela rialzista che ingloba completamente il corpo della candela ribassista precedente
    """
    prev_body_top = max(prev['open'], prev['close'])
    prev_body_bottom = min(prev['open'], prev['close'])
    curr_body_top = max(curr['open'], curr['close'])
    curr_body_bottom = min(curr['open'], curr['close'])
    
    # Prev deve essere ribassista, curr rialzista
    is_prev_bearish = prev['close'] < prev['open']
    is_curr_bullish = curr['close'] > curr['open']
    
    # Curr deve inglobare il corpo di prev
    engulfs = (curr_body_bottom <= prev_body_bottom and 
               curr_body_top >= prev_body_top)
    
    # Curr deve avere un corpo decente (almeno 50% del range di prev)
    prev_body = abs(prev['open'] - prev['close'])
    curr_body = abs(curr['open'] - curr['close'])
    has_body = curr_body >= prev_body * 0.5
    
    return is_prev_bearish and is_curr_bullish and engulfs and has_body


def is_bearish_engulfing(prev, curr):
    """
    Pattern: Bearish Engulfing
    Candela ribassista che ingloba completamente il corpo della candela rialzista precedente
    """
    prev_body_top = max(prev['open'], prev['close'])
    prev_body_bottom = min(prev['open'], prev['close'])
    curr_body_top = max(curr['open'], curr['close'])
    curr_body_bottom = min(curr['open'], curr['close'])
    
    # Prev deve essere rialzista, curr ribassista
    is_prev_bullish = prev['close'] > prev['open']
    is_curr_bearish = curr['close'] < curr['open']
    
    # Curr deve inglobare il corpo di prev
    engulfs = (curr_body_top >= prev_body_top and 
               curr_body_bottom <= prev_body_bottom)
    
    # Curr deve avere un corpo decente
    prev_body = abs(prev['open'] - prev['close'])
    curr_body = abs(curr['open'] - curr['close'])
    has_body = curr_body >= prev_body * 0.5
    
    return is_prev_bullish and is_curr_bearish and engulfs and has_body


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


def is_morning_star(a, b, c):
    """
    Pattern: Morning Star (3 candele - bullish reversal)
    1. Candela ribassista grande
    2. Candela piccola (indecisione)
    3. Candela rialzista grande che recupera oltre il 50%
    """
    # Prima candela: ribassista con corpo decente
    a_body = abs(a['close'] - a['open'])
    a_range = a['high'] - a['low']
    is_a_bearish = a['close'] < a['open']
    
    # Seconda candela: piccola (corpo < 30% della prima)
    b_body = abs(b['close'] - b['open'])
    is_b_small = b_body < a_body * 0.3
    
    # Terza candela: rialzista e chiude sopra il 50% della prima
    is_c_bullish = c['close'] > c['open']
    c_recovers = c['close'] > (a['open'] + a['close']) / 2
    
    return (is_a_bearish and 
            a_body / a_range > 0.5 and  # Prima candela ha corpo significativo
            is_b_small and 
            is_c_bullish and 
            c_recovers)


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
    if not is_morning_star(a, b, c):
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
    all_above_ema60 = (cons_lows > ema60_during_cons).all()
    
    if not all_above_ema60:
        logging.debug(f'🚫 Triple Touch: Consolidamento rompe sotto EMA 60')
        return (False, None)
    
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
    if vol_ratio < 2.0:
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
            
            if vol_ratio < 2.0:
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
    
    if recovery_pct < 0.80:
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
    if vol_ratio < 2.0:
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


def is_pin_bar(candle):
    """
    Pattern: Pin Bar
    Ombra molto lunga da un lato, corpo piccolo
    """
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    
    if total_range == 0:
        return False
    
    upper_wick = candle['high'] - max(candle['close'], candle['open'])
    lower_wick = min(candle['close'], candle['open']) - candle['low']
    
    # Pin bar rialzista: ombra inferiore lunga
    bullish_pin = (lower_wick >= total_range * 0.6 and 
                   body <= total_range * 0.3)
    
    # Pin bar ribassista: ombra superiore lunga
    bearish_pin = (upper_wick >= total_range * 0.6 and 
                   body <= total_range * 0.3)
    
    return bullish_pin or bearish_pin


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
    if vol_ratio < 1.8:
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
    for flag_duration in range(3, 9):  # 3, 4, 5, 6, 7, 8
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
                    volume_ok = (vol_ratio > 2.0 and 
                                breakout_vol > pole_vol * 0.6)
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


def check_patterns(df: pd.DataFrame):
    """
    Controlla pattern con volume check adaptive
    
    Args:
        df: DataFrame OHLCV
        symbol: Symbol name (per auto-discovery check)
    """
    if len(df) < 6:
        return (False, None, None, None)
    
    # === FILTRO VOLUME (ADAPTIVE) ===
    if VOLUME_FILTER_ENABLED:
        # Check se auto-discovered
        with AUTO_DISCOVERED_LOCK:
            is_auto = symbol in AUTO_DISCOVERED_SYMBOLS if symbol else False
        
        # Passa symbol e flag a volume_confirmation
        vol_ok = volume_confirmation(
            df, 
            min_ratio=VOLUME_THRESHOLDS.get(VOLUME_FILTER_MODE, 1.5),
            symbol=symbol,
            is_auto_discovered=is_auto
        )
        
        if not vol_ok:
            logging.info(f'❌ {symbol}: Pattern search BLOCKED by volume filter')
            return (False, None, None, None)
    
    # Filtro Trend - MUST PASS
    if not is_uptrend_structure(df):
        logging.info('❌ Pattern search BLOCKED: No uptrend structure')
        return (False, None, None, None)
    
    # Filtro ATR - WARNING (non blocking ma logged)
    if not atr_expanding(df):
        logging.info('⚠️ ATR not expanding - pattern may be less reliable')
    
    # ===== TIER 1: HIGH PROBABILITY PATTERNS =====
    
    # 🥇 PRIORITY #1: Volume Spike Breakout
    if AVAILABLE_PATTERNS.get('volume_spike_breakout', {}).get('enabled', False):
        found, data = is_volume_spike_breakout(df)
        if found:
            logging.info(f'✅ TIER 1 Pattern: Volume Spike Breakout (volume: {data["volume_ratio"]:.1f}x)')
            return (True, 'Buy', 'Volume Spike Breakout', data)

    # 🥇 BREAKOUT + RETEST (NUOVO - inserisci qui!)
    if AVAILABLE_PATTERNS.get('breakout_retest', {}).get('enabled', False):
        found, data = is_breakout_retest(df)
        if found:
            logging.info(
                f'✅ TIER 1 Pattern: Breakout + Retest '
                f'(range: {data["range_pct"]:.2f}%, '
                f'rejection: {data["retest_rejection_pct"]:.1f}%)'
            )
            return (True, 'Buy', 'Breakout + Retest', data)

    # 🥇 TRIPLE TOUCH BREAKOUT (NUOVO)
    if AVAILABLE_PATTERNS.get('triple_touch_breakout', {}).get('enabled', False):
        found, data = is_triple_touch_breakout(df)
        if found:
            logging.info(
                f'✅ TIER 1 Pattern: Triple Touch Breakout '
                f'(R: ${data["resistance"]:.4f}, '
                f'vol: {data["volume_ratio"]:.1f}x, '
                f'quality: {data["quality"]})'
            )
            return (True, 'Buy', 'Triple Touch Breakout', data)

    # 👇 LIQUIDITY SWEEP (massima priorità - pattern istituzionale)
    if AVAILABLE_PATTERNS['liquidity_sweep_reversal']['enabled']:
        found, sweep_data = is_liquidity_sweep_reversal(df)
        if found:
            return (True, 'Buy', 'Liquidity Sweep + Reversal', sweep_data)

    # 3. Support/Resistance Bounce
    if AVAILABLE_PATTERNS['sr_bounce']['enabled']:
        found, sr_data = is_support_resistance_bounce(df)
        if found:
            return (True, 'Buy', 'Support/Resistance Bounce', sr_data)
    
    # ===== TIER 2: EXISTING PATTERNS (Lower Priority) =====
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Bullish Comeback
    if AVAILABLE_PATTERNS['bullish_comeback']['enabled']:
        if is_bullish_comeback(df):
            return (True, 'Buy', 'Bullish Comeback', None)
    
    # Compression Breakout (Enhanced)
    if AVAILABLE_PATTERNS['compression_breakout']['enabled']:
        if is_compression_breakout(df):
            # === ENHANCEMENT: HTF RESISTANCE CHECK ===
            # Nota: Serve symbol e timeframe dal contesto
            # Questa logica va in analyze_job() quando pattern è trovato
            # Per ora ritorna pattern trovato
            return (True, 'Buy', 'Compression Breakout (Enhanced)', None)
    
    # Bullish Flag Breakout (CON DATI CUSTOM)
    if AVAILABLE_PATTERNS['bullish_flag_breakout']['enabled']:
        found, flag_data = is_bullish_flag_breakout(df)
        if found:
            return (True, 'Buy', 'Bullish Flag Breakout', flag_data)

    # 👇 AGGIUNGI QUI - Morning Star + EMA Breakout (alta priorità!)
    if AVAILABLE_PATTERNS['morning_star_ema_breakout']['enabled']:
        if is_morning_star_ema_breakout(df):
            return (True, 'Buy', 'Morning Star + EMA Breakout', None)
    
    # Bullish Engulfing
    if AVAILABLE_PATTERNS['bullish_engulfing']['enabled']:
        if is_bullish_engulfing(prev, last):
            return (True, 'Buy', 'Bullish Engulfing', None)
    
    # Bearish Engulfing (SELL)
    if AVAILABLE_PATTERNS['bearish_engulfing']['enabled']:
        if is_bearish_engulfing(prev, last):
            return (True, 'Sell', 'Bearish Engulfing', None)
    
    # Hammer
    if AVAILABLE_PATTERNS['hammer']['enabled']:
        if is_hammer(last):
            return (True, 'Buy', 'Hammer', None)
    
    # Shooting Star (SELL)
    if AVAILABLE_PATTERNS['shooting_star']['enabled']:
        if is_shooting_star(last):
            return (True, 'Sell', 'Shooting Star', None)
    
    # Pin bar
    if AVAILABLE_PATTERNS['pin_bar_bullish']['enabled'] or AVAILABLE_PATTERNS['pin_bar_bearish']['enabled']:
        if is_pin_bar(last):
            lower_wick = min(last['open'], last['close']) - last['low']
            upper_wick = last['high'] - max(last['open'], last['close'])
            
            if lower_wick > upper_wick:
                # Pin bar bullish
                if AVAILABLE_PATTERNS['pin_bar_bullish']['enabled']:
                    return (True, 'Buy', 'Pin Bar Bullish', None)
            else:
                # Pin bar bearish
                if AVAILABLE_PATTERNS['pin_bar_bearish']['enabled']:
                    return (True, 'Sell', 'Pin Bar Bearish', None)
    
    # Doji
    if AVAILABLE_PATTERNS['doji']['enabled']:
        if is_doji(last):
            # Il doji può essere sia BUY che SELL a seconda del trend
            if prev['close'] > prev['open']:
                return (True, 'Sell', 'Doji (reversione)', None)
            else:
                return (True, 'Buy', 'Doji (reversione)', None)
    
    # Morning Star
    if AVAILABLE_PATTERNS['morning_star']['enabled']:
        if is_morning_star(prev2, prev, last):
            return (True, 'Buy', 'Morning Star', None)
    
    # Evening Star (SELL)
    if AVAILABLE_PATTERNS['evening_star']['enabled']:
        if is_evening_star(prev2, prev, last):
            return (True, 'Sell', 'Evening Star', None)
    
    # Three White Soldiers
    if AVAILABLE_PATTERNS['three_white_soldiers']['enabled']:
        if is_three_white_soldiers(prev2, prev, last):
            return (True, 'Buy', 'Three White Soldiers', None)
    
    # Three Black Crows (SELL)
    if AVAILABLE_PATTERNS['three_black_crows']['enabled']:
        if is_three_black_crows(prev2, prev, last):
            return (True, 'Sell', 'Three Black Crows', None)
    
    return (False, None, None, None)


async def get_open_positions_from_bybit(symbol: str = None):
    """
    Recupera le posizioni aperte reali da Bybit
    Se symbol è specificato, controlla solo quel symbol
    Altrimenti ritorna tutte le posizioni
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
                        'entry_price': float(pos.get('avgPrice', 0)),
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
        
        with POSITIONS_LOCK:
            # Crea un set dei symbol con posizioni reali
            real_symbols = {pos['symbol'] for pos in real_positions}
            
            # Rimuovi dal tracking le posizioni che non esistono più su Bybit
            to_remove = []
            for symbol in ACTIVE_POSITIONS.keys():
                if symbol not in real_symbols:
                    to_remove.append(symbol)
            
            for symbol in to_remove:
                logging.info(f'🔄 Rimossa {symbol} dal tracking (non presente su Bybit)')
                del ACTIVE_POSITIONS[symbol]
            
            # Aggiungi al tracking posizioni che esistono su Bybit ma non sono tracciate
            for pos in real_positions:
                symbol = pos['symbol']
                if symbol not in ACTIVE_POSITIONS:
                    ACTIVE_POSITIONS[symbol] = {
                        'side': pos['side'],
                        'qty': pos['size'],
                        'sl': 0,  # Non disponibile da API posizioni
                        'tp': 0,  # Non disponibile da API posizioni
                        'order_id': None,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'synced_from_bybit': True
                    }
                    logging.info(f'🔄 Aggiunta {symbol} al tracking (trovata su Bybit)')
        
        logging.info(f'✅ Sync posizioni completata: {len(real_positions)} posizioni attive')
        return True
        
    except Exception as e:
        logging.exception('Errore in sync_positions_with_bybit')
        return False


def calculate_position_size(entry_price: float, sl_price: float, risk_usd: float):
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
    
    # ===== CORREZIONE LOGICA =====
    # Check resistenza (EMA SOTTO il prezzo = resistenza!)
    if current_tf in ['5m', '15m']:
        # Per scalping: controlla EMA 5 e 10 su 30m
        # BLOCCA se EMA 5 o 10 sono SOTTO il prezzo corrente (resistenza sopra)
        if ema5_htf > current_price or ema10_htf > current_price:  # 👈 CORRETTO
            # EMA sopra = resistenza
            return {
                'blocked': True,
                'htf': htf,
                'details': f'EMA 5 ({htf}): ${ema5_htf:.2f}\nEMA 10 ({htf}): ${ema10_htf:.2f}\nPrice: ${current_price:.2f}'
            }
    
    elif current_tf in ['30m', '1h']:
        # Per day: controlla EMA 60 su 4h
        ema60_htf = df_htf['close'].ewm(span=60, adjust=False).mean().iloc[-1]
        
        # BLOCCA se EMA 60 è SOPRA il prezzo
        if ema60_htf > current_price:  # 👈 CORRETTO
            return {
                'blocked': True,
                'htf': htf,
                'details': f'EMA 60 ({htf}): ${ema60_htf:.2f}\nPrice: ${current_price:.2f}'
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


async def place_bybit_order(symbol: str, side: str, qty: float, sl_price: float, tp_price: float, entry_price: float, timeframe: str, chat_id: int):
    """
    Piazza ordine market su Bybit (Demo o Live)
    Controlla REALMENTE su Bybit se esiste già una posizione aperta
    
    Parametri:
    - symbol: es. 'BTCUSDT'
    - side: 'Buy' o 'Sell'
    - qty: quantità in contratti
    - sl_price: prezzo stop loss
    - tp_price: prezzo take profit
    """
    if BybitHTTP is None:
        return {'error': 'pybit non disponibile'}
    
    # SINCRONIZZA con Bybit prima di controllare
    await sync_positions_with_bybit()
    
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
            instrument_info = session.get_instruments_info(
                category='linear',
                symbol=symbol
            )
            
            if instrument_info.get('retCode') == 0:
                instruments = instrument_info.get('result', {}).get('list', [])
                if instruments:
                    lot_size_filter = instruments[0].get('lotSizeFilter', {})
                    min_order_qty = float(lot_size_filter.get('minOrderQty', 0.001))
                    max_order_qty = float(lot_size_filter.get('maxOrderQty', 1000000))
                    qty_step = float(lot_size_filter.get('qtyStep', 0.001))
                    
                    logging.info(f'📊 {symbol} - Min: {min_order_qty}, Max: {max_order_qty}, Step: {qty_step}')
                    
                    # Arrotonda qty al qty_step più vicino
                    qty = round(qty / qty_step) * qty_step
                    
                    # Limita qty tra min e max
                    qty = max(min_order_qty, min(qty, max_order_qty))
                    
                    # Assicurati che qty rispetti il formato
                    # Conta decimali in qty_step
                    decimals = len(str(qty_step).split('.')[-1]) if '.' in str(qty_step) else 0
                    qty = round(qty, decimals)
                else:
                    logging.warning(f'Nessuna info trovata per {symbol}, uso default')
                    qty = round(qty, 3)
            else:
                logging.warning(f'Errore nel recuperare info {symbol}, uso default')
                qty = round(qty, 3)
                
        except Exception as e:
            logging.warning(f'Errore nel recuperare instrument info: {e}')
            # Fallback: arrotondamento generico
            qty = round(qty, 3)
        
        # Verifica qty minima sensata
        if qty < 0.001:
            return {'error': f'Qty troppo piccola ({qty}). Aumenta RISK_USD o riduci ATR.'}
        
        # Arrotonda prezzi con decimali dinamici
        price_decimals = get_price_decimals(sl_price)
        sl_price = round(sl_price, price_decimals)
        tp_price = round(tp_price, price_decimals)
        
        logging.info(f'📤 Piazzando ordine {side} per {symbol}')
        logging.info(f'   Qty: {qty} | SL: {sl_price} | TP: {tp_price}')
        
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
        
        logging.info(f'✅ Ordine eseguito: {order}')
        
        # Salva la posizione come attiva
        if order.get('retCode') == 0:
            with POSITIONS_LOCK:
                ACTIVE_POSITIONS[symbol] = {
                    'side': side,
                    'qty': qty,
                    'entry_price': entry_price,  # 👈 AGGIUNGI (pass come parametro)
                    'sl': sl_price,
                    'tp': tp_price,
                    'order_id': order.get('result', {}).get('orderId'),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'timeframe': timeframe,  # 👈 AGGIUNGI (pass come parametro)
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
        url = f'{BYBIT_PUBLIC_REST}/v5/market/tickers'
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
            if symbol in AUTO_DISCOVERY_CONFIG['exclude_symbols']:
                continue
            
            try:
                volume_24h = float(ticker.get('turnover24h', 0))  # Volume in USDT
                price_change_percent = float(ticker.get('price24hPcnt', 0)) * 100  # In percentuale
                last_price = float(ticker.get('lastPrice', 0))
                
                # Applica filtri
                if volume_24h < AUTO_DISCOVERY_CONFIG['min_volume_usdt']:
                    continue
                
                if price_change_percent < AUTO_DISCOVERY_CONFIG['min_price_change']:
                    continue
                
                if price_change_percent > AUTO_DISCOVERY_CONFIG['max_price_change']:
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
        sort_by = AUTO_DISCOVERY_CONFIG['sorting']
        
        if sort_by == 'volume':
            candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
        else:  # price_change_percent (default)
            candidates.sort(key=lambda x: x['price_change_percent'], reverse=True)
        
        # Prendi top N
        top_count = AUTO_DISCOVERY_CONFIG['top_count']
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
    if not AUTO_DISCOVERY_CONFIG['enabled']:
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
        
        timeframe = AUTO_DISCOVERY_CONFIG['timeframe']
        autotrade = AUTO_DISCOVERY_CONFIG['autotrade']
        
        # Converti in set per comparazione
        new_symbols_set = set(top_symbols)
        
        with AUTO_DISCOVERED_LOCK:
            old_symbols_set = set(AUTO_DISCOVERED_SYMBOLS)
        
        # Symbols da rimuovere (non più in top)
        to_remove = old_symbols_set - new_symbols_set
        
        # Symbols da aggiungere (nuovi in top)
        to_add = new_symbols_set - old_symbols_set
        
        # === RIMUOVI ANALISI VECCHIE ===
        removed_count = 0
        
        with ACTIVE_ANALYSES_LOCK:
            chat_analyses = ACTIVE_ANALYSES.get(chat_id, {})
            
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
            with ACTIVE_ANALYSES_LOCK:
                chat_map = ACTIVE_ANALYSES.setdefault(chat_id, {})
                
                if key in chat_map:
                    logging.debug(f'⏭️ Skip {symbol}: già in analisi')
                    continue
            
            # Verifica dati disponibili
            test_df = bybit_get_klines(symbol, timeframe, limit=10)
            if test_df.empty:
                logging.warning(f'⚠️ Skip {symbol}: nessun dato disponibile')
                continue
            
            # Calcola intervallo
            interval_seconds = INTERVAL_SECONDS.get(timeframe, 300)
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
            
            with ACTIVE_ANALYSES_LOCK:
                chat_map[key] = job
            
            added_count += 1
            logging.info(f'✅ Aggiunto {symbol} {timeframe}')
        
        # Aggiorna storage
        with AUTO_DISCOVERED_LOCK:
            AUTO_DISCOVERED_SYMBOLS.clear()
            AUTO_DISCOVERED_SYMBOLS.update(new_symbols_set)
        
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
            msg += f"🔄 Prossimo update tra 4 ore"
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
            

async def update_trailing_stop_loss(context: ContextTypes.DEFAULT_TYPE):
    """
    Job che aggiorna trailing stop loss per tutte le posizioni attive
    Viene eseguito ogni 5 minuti
    
    LOGICA:
    1. Per ogni posizione aperta
    2. Scarica dati timeframe EMA di riferimento
    3. Calcola EMA 10
    4. Se prezzo > entry + activation_percent:
       - Attiva trailing
       - Sposta SL sotto EMA 10 (mai indietro)
    5. Aggiorna SL su Bybit
    """
    if not TRAILING_STOP_ENABLED:
        return
    
    with POSITIONS_LOCK:
        positions_copy = dict(ACTIVE_POSITIONS)
    
    if not positions_copy:
        logging.debug('🔇 Nessuna posizione attiva per trailing SL')
        return
    
    logging.info(f'🔄 Checking trailing SL per {len(positions_copy)} posizioni...')
    
    for symbol, pos_info in positions_copy.items():
        try:
            # Solo posizioni BUY (per ora)
            if pos_info['side'] != 'Buy':
                continue
            
            # Dati posizione
            entry_price = pos_info['entry_price']
            current_sl = pos_info['sl']
            timeframe_entry = pos_info['timeframe']
            trailing_active = pos_info.get('trailing_active', False)
            highest_price = pos_info.get('highest_price', entry_price)
            
            # Determina TF per EMA trailing
            ema_tf = TRAILING_EMA_TIMEFRAME.get(timeframe_entry, timeframe_entry)
            
            # Scarica dati
            df = bybit_get_klines(symbol, ema_tf, limit=50)
            if df.empty:
                logging.warning(f'⚠️ Nessun dato per {symbol} {ema_tf} (trailing SL)')
                continue
            
            current_price = df['close'].iloc[-1]
            
            # Aggiorna highest price
            if current_price > highest_price:
                highest_price = current_price
                with POSITIONS_LOCK:
                    if symbol in ACTIVE_POSITIONS:
                        ACTIVE_POSITIONS[symbol]['highest_price'] = highest_price
            
            # Calcola profit %
            profit_percent = ((current_price - entry_price) / entry_price) * 100
            
            # Check se attivare trailing
            if not trailing_active and profit_percent >= TRAILING_CONFIG['activation_percent']:
                logging.info(f'✅ Trailing SL ATTIVATO per {symbol} (profit: {profit_percent:.2f}%)')
                with POSITIONS_LOCK:
                    if symbol in ACTIVE_POSITIONS:
                        ACTIVE_POSITIONS[symbol]['trailing_active'] = True
                trailing_active = True
            
            # Se trailing non attivo, skip
            if not trailing_active:
                logging.debug(f'⏳ {symbol}: Trailing non ancora attivo (profit: {profit_percent:.2f}%)')
                continue
            
            # Calcola EMA 10
            ema_10 = df['close'].ewm(span=10, adjust=False).mean().iloc[-1]
            
            # Nuovo SL: sotto EMA 10 con buffer
            new_sl = ema_10 * (1 - TRAILING_CONFIG['ema_buffer'])
            
            # NEVER BACK: SL non torna mai indietro
            if TRAILING_CONFIG['never_back'] and new_sl <= current_sl:
                logging.debug(f'🔒 {symbol}: SL rimane {current_sl:.4f} (new {new_sl:.4f} è minore)')
                continue
            
            # Verifica che nuovo SL non sia sopra il prezzo corrente (safety)
            if new_sl >= current_price * 0.998:  # Almeno 0.2% sotto
                logging.warning(f'⚠️ {symbol}: New SL {new_sl:.4f} troppo vicino a price {current_price:.4f}')
                continue
            
            # Arrotonda con decimali dinamici
            price_decimals = get_price_decimals(new_sl)
            new_sl = round(new_sl, price_decimals)
            
            # Differenza significativa? (almeno 0.1% di movimento)
            sl_move_percent = ((new_sl - current_sl) / current_sl) * 100
            if sl_move_percent < 0.1:
                logging.debug(f'🔹 {symbol}: SL move troppo piccolo ({sl_move_percent:.2f}%)')
                continue
            
            logging.info(f'🔼 {symbol}: Aggiornamento SL da {current_sl:.{price_decimals}f} a {new_sl:.{price_decimals}f}')
            logging.info(f'   Price: {current_price:.{price_decimals}f}, EMA 10 ({ema_tf}): {ema_10:.{price_decimals}f}')

            # ===== VERIFICA POSIZIONE ESISTE SU BYBIT =====
            if BybitHTTP is not None:
                try:
                    session = create_bybit_session()
                    
                    # 🔧 FIX: Definisci 'positions' PRIMA di usarlo
                    positions = session.get_positions(
                        category='linear',
                        symbol=symbol
                    )
                    
                    if positions.get('retCode') == 0:
                        pos_list = positions.get('result', {}).get('list', [])
                        
                        # Verifica se c'è una posizione con size > 0
                        position_exists = False
                        for pos in pos_list:
                            if float(pos.get('size', 0)) > 0:
                                position_exists = True
                                break
                        
                        if not position_exists:
                            logging.warning(f'⚠️ {symbol}: Posizione NON trovata su Bybit, rimuovo dal tracking')
                            with POSITIONS_LOCK:
                                if symbol in ACTIVE_POSITIONS:
                                    del ACTIVE_POSITIONS[symbol]
                            continue
                    else:
                        # Errore API nella verifica posizione
                        logging.error(f'❌ {symbol}: Errore verifica posizione: {positions.get("retMsg")}')
                        continue
                            
                    # Aggiorna SL solo se posizione esiste
                    result = session.set_trading_stop(
                        category='linear',
                        symbol=symbol,
                        positionIdx=0,  # One-way mode
                        stopLoss=str(new_sl)
                    )
                    
                    if result.get('retCode') == 0:
                        # Aggiorna tracking locale
                        with POSITIONS_LOCK:
                            if symbol in ACTIVE_POSITIONS:
                                ACTIVE_POSITIONS[symbol]['sl'] = new_sl
                        
                        logging.info(f'✅ {symbol}: Trailing SL aggiornato su Bybit a ${new_sl:.{price_decimals}f}')

                        # ===== INVIA NOTIFICA TELEGRAM =====
                        chat_id = pos_info.get('chat_id')
                        
                        if chat_id:
                            try:
                                # Calcola profit attuale
                                profit_percent = ((current_price - entry_price) / entry_price) * 100
                                profit_usd = (current_price - entry_price) * pos_info['qty']
                                
                                # Calcola quanto si è spostato lo SL
                                sl_move_usd = (new_sl - current_sl) * pos_info['qty']
                                
                                notification = (
                                    f"🔼 <b>Trailing Stop Aggiornato</b>\n\n"
                                    f"🪙 <b>Symbol:</b> {symbol} ({timeframe_entry})\n"
                                    f"💵 <b>Prezzo:</b> ${current_price:.{price_decimals}f}\n"
                                    f"📈 <b>Profit:</b> {profit_percent:+.2f}% (${profit_usd:+.2f})\n\n"
                                    f"🛑 <b>Stop Loss:</b>\n"
                                    f"  Prima: ${current_sl:.{price_decimals}f}\n"
                                    f"  Ora: ${new_sl:.{price_decimals}f}\n"
                                    f"  Spostamento: +${sl_move_usd:.2f}\n\n"
                                    f"📊 <b>EMA 10 ({ema_tf}):</b> ${ema_10:.{price_decimals}f}\n"
                                    f"💡 SL protegge ora ${(new_sl - entry_price) * pos_info['qty']:.2f} di profit"
                                )
                                
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=notification,
                                    parse_mode='HTML'
                                )
                                
                                logging.info(f'📱 Notifica trailing inviata per {symbol}')
                            
                            except Exception as e:
                                logging.error(f'❌ Errore invio notifica trailing: {e}')
                    
                    else:
                        logging.error(f'❌ {symbol}: Errore aggiornamento SL Bybit: {result.get("retMsg")}')
                
                except Exception as e:
                    logging.error(f'❌ {symbol}: Errore set_trading_stop: {e}')
            
        except Exception as e:
            logging.exception(f'❌ Errore trailing SL per {symbol}: {e}')


# ===== FUNZIONE per schedulare il job =====

def schedule_trailing_stop_job(application):
    """
    Schedula il job di trailing stop loss ogni 5 minuti
    """
    if not TRAILING_STOP_ENABLED:
        logging.info('🔕 Trailing Stop Loss disabilitato')
        return
    
    interval = TRAILING_CONFIG['check_interval']
    
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

    # Check se auto-discovered
    is_auto = job_ctx.get('auto_discovered', False)

    # Verifica se le notifiche complete sono attive
    with FULL_NOTIFICATIONS_LOCK:
        full_mode = chat_id in FULL_NOTIFICATIONS and key in FULL_NOTIFICATIONS[chat_id]

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

        last_close = df['close'].iloc[-1]
        last_time = df.index[-1]
        timestamp_str = last_time.strftime('%Y-%m-%d %H:%M UTC')

        # ===== CALCOLA DECIMALI UNA SOLA VOLTA =====
        price_decimals = get_price_decimals(last_close)
        
        # ===== STEP 2: PRE-FILTER EMA (PRIMA DI CERCARE PATTERN) =====
        ema_analysis = None
        pattern_search_allowed = True  # Default: cerca pattern
        
        if EMA_FILTER_ENABLED:
            ema_analysis = analyze_ema_conditions(df, timeframe, None)
            
            logging.info(
                f'📊 EMA Pre-Filter {symbol} {timeframe}: '
                f'Score={ema_analysis["score"]}, '
                f'Quality={ema_analysis["quality"]}, '
                f'Passed={ema_analysis["passed"]}'
            )
            
            # STRICT MODE: Blocca completamente se EMA non passa
            if EMA_FILTER_MODE == 'strict' and not ema_analysis['passed']:
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
                
                return  # STOP QUI - Non cerca pattern
            
            # LOOSE MODE: Blocca se score < 40
            elif EMA_FILTER_MODE == 'loose' and not ema_analysis['passed']:
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
                
                return  # STOP - Non cerca pattern
        
        # ===== STEP 3: CERCA PATTERN (solo se EMA permette) =====
        found = False
        side = None
        pattern = None
        pattern_data = None
        
        if pattern_search_allowed:
            found, side, pattern, pattern_data = check_patterns(df, symbol=symbol)
            
            if found:
                logging.info(f'🎯 Pattern trovato: {pattern} ({side}) su {symbol} {timeframe}')
            else:
                logging.info(f'🔍 {symbol} {timeframe} - Nessun pattern rilevato')
        
        # Se NON pattern e NON full_mode → Skip notifica
        if not found and not full_mode:
            logging.debug(f'🔕 {symbol} {timeframe} - No pattern, no full mode → Skip')
            return
        
        # ===== STEP 4: CALCOLA PARAMETRI TRADING =====
        atr_series = atr(df, period=14)
        last_atr = atr_series.iloc[-1] if not atr_series.isna().all() else np.nan
        
        # ===== STEP 5: COSTRUISCI MESSAGGIO =====
        
        if found and side == 'Buy':
            # Check Higher Timeframe EMA (tappo)
            htf_block = check_higher_timeframe_resistance(
                symbol=symbol,
                current_tf=timeframe,
                current_price=last_close
            )
            
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

            # ===== GESTIONE PATTERN-SPECIFIC ENTRY/SL/TP =====
            
            # === BULLISH FLAG BREAKOUT ===
            if pattern == 'Bullish Flag Breakout' and pattern_data:
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

            if pattern == 'Triple Touch Breakout' and pattern_data:
                """
                ENTRY LOGIC per Triple Touch Breakout
                
                Entry: Al breakout del terzo tocco (prezzo corrente)
                SL: Sotto consolidamento low (con buffer 0.2%)
                TP: R + (2.5 × range consolidamento)
                
                R:R tipico: 1:2.5-3.5
                """
                
                entry_price = pattern_data['suggested_entry']
                sl_price = pattern_data['suggested_sl']
                tp_price = pattern_data['suggested_tp']
                
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

            if pattern == 'Breakout + Retest' and pattern_data:
                """
                ENTRY LOGIC per Breakout + Retest
                
                Entry: Al bounce dal retest (prezzo corrente)
                SL: Sotto retest low (con buffer 0.2%)
                TP: Resistance + (2 × range consolidamento)
                
                R:R tipico: 1:2.5-3
                """
                
                entry_price = pattern_data['suggested_entry']
                sl_price = pattern_data['suggested_sl']
                tp_price = pattern_data['suggested_tp']
                
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
            risk_for_symbol = SYMBOL_RISK_OVERRIDE.get(symbol, RISK_USD)
            qty = calculate_position_size(entry_price, sl_price, risk_for_symbol)
            position_exists = symbol in ACTIVE_POSITIONS
            
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
                if USE_EMA_STOP_LOSS:
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
            
            # Volume
            if VOLUME_FILTER:
                vol = df['volume']
                if len(vol) >= 21:
                    avg_vol = vol.iloc[-21:-1].mean()
                    current_vol = vol.iloc[-1]
                    vol_ratio = (current_vol / avg_vol) if avg_vol > 0 else 0
                    caption += f"📊 <b>Volume:</b> {vol_ratio:.2f}x\n"
            
            # EMA Analysis dettagliata
            if ema_analysis:
                # Logica speciale per Liquidity Sweep
                if pattern == 'Liquidity Sweep + Reversal':
                    ema_analysis = analyze_ema_conditions(df, timeframe, pattern)
                
                caption += "\n━━━━━━━━━━━━━━━━━━━\n"
                caption += "📈 <b>EMA Analysis</b>\n\n"
                caption += ema_analysis['details']
                
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
            
            # Warning se LOOSE mode con EMA deboli
            if ema_analysis and EMA_FILTER_MODE == 'loose' and not ema_analysis['passed']:
                caption += f"\n\n⚠️ <b>ATTENZIONE:</b> Setup con EMA non ottimali"
                caption += f"\nConsidera ridurre position size o aspettare."
            
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
                    chat_id
                )
                
                if 'error' in order_res:
                    caption += f"\n\n❌ <b>Errore ordine:</b>\n{order_res['error']}"
                else:
                    caption += f"\n\n✅ <b>Ordine su Bybit {TRADING_MODE.upper()}</b>"
        
        elif found and side == 'Sell':
            # SEGNALE SELL (se abilitato)
            caption = f"🔴 <b>SEGNALE SELL</b>\n\n"
            caption += f"📊 Pattern: {pattern}\n"
            caption += f"🪙 {symbol} ({timeframe})\n\n"
            caption += "⚠️ Pattern SELL rilevato ma NON tradato\n"
            caption += "(Solo pattern BUY sono attivi)"
        
        else:
            # NESSUN PATTERN (full mode)
            caption = f"📊 <b>{symbol}</b> ({timeframe})\n"
            caption += f"🕐 {timestamp_str}\n"
            caption += f"💵 Price: ${last_close:.{price_decimals}f}\n\n"
            
            # INFO FILTRI GLOBALI CON GESTIONE ERRORI
            caption += "🔍 <b>Global Filters Status:</b>\n"
            
            # Check volume CON error handling
            vol_ratio = 0.0
            vol_result = False
            
            try:
                if 'volume' in df.columns and len(df['volume']) >= 20:
                    vol = df['volume']
                    avg_vol = vol.iloc[-20:-1].mean()
                    current_vol = vol.iloc[-1]
                    
                    if avg_vol > 0 and not pd.isna(avg_vol) and not pd.isna(current_vol):
                        vol_ratio = current_vol / avg_vol
                        vol_result = vol_ratio > 1.5
            except Exception as e:
                logging.error(f'Error calculating volume ratio: {e}')
            
            if vol_result:
                caption += f"✅ Volume: {vol_ratio:.1f}x (>1.5x) OK\n"
            else:
                if vol_ratio > 0:
                    caption += f"❌ Volume: {vol_ratio:.1f}x (serve >1.5x) BLOCKED\n"
                else:
                    caption += f"❌ Volume: N/A (dati insufficienti) BLOCKED\n"
            
            # Check uptrend
            try:
                trend_result = is_uptrend_structure(df)
                if trend_result:
                    caption += "✅ Uptrend Structure: HH+HL OK\n"
                else:
                    caption += "❌ Uptrend Structure: NO\n"
            except Exception as e:
                logging.error(f'Error checking uptrend: {e}')
                caption += "❌ Uptrend Structure: ERROR\n"
            
            # Check ATR
            try:
                atr_result = atr_expanding(df)
                if atr_result:
                    caption += "✅ ATR Expansion: Volatilità in aumento\n"
                else:
                    caption += "⚠️ ATR Flat: Bassa volatilità\n"
            except Exception as e:
                logging.error(f'Error checking ATR: {e}')
                caption += "⚠️ ATR: ERROR\n"
            
            caption += "\n"
            
            # Pattern search status
            if not vol_result or not trend_result:
                caption += "🚫 <b>Pattern search bloccata</b>\n"
            else:
                caption += "🔔 <b>Full Mode - Nessun pattern rilevato</b>\n"
            
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
            with FULL_NOTIFICATIONS_LOCK:
                should_send = chat_id in FULL_NOTIFICATIONS and key in FULL_NOTIFICATIONS[chat_id]
            
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
    mode_emoji = "🎮" if TRADING_MODE == 'demo' else "💰"
    mode_text = "DEMO (fondi virtuali)" if TRADING_MODE == 'demo' else "LIVE (SOLDI REALI!)"
    
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start"""
    bot_username = (await context.bot.get_me()).username
    
    # Emoji per la modalità
    mode_emoji = "🎮" if TRADING_MODE == 'demo' else "💰"
    mode_text = "DEMO (fondi virtuali)" if TRADING_MODE == 'demo' else "LIVE (SOLDI REALI!)"
    
    welcome_text = (
        f"🤖 <b>Bot Pattern Detection Attivo!</b>\n"
        f"👤 Username: @{bot_username}\n"
        f"{mode_emoji} <b>Modalità: {mode_text}</b>\n\n"
        "📊 <b>Comandi Analisi:</b>\n"
        "/analizza SYMBOL TF [autotrade] - Inizia analisi\n"
        "/stop SYMBOL - Ferma analisi\n"
        "/list - Analisi attive\n"
        "/abilita SYMBOL TF - Notifiche complete\n"
        "/pausa SYMBOL TF - Solo pattern (default)\n"
        "/test SYMBOL TF - Test pattern\n\n"
        "💼 <b>Comandi Trading:</b>\n"
        "/balance - Mostra saldo\n"
        "/posizioni - Posizioni aperte\n"
        "/orders [N] - Ultimi ordini e P&L\n"
        "/trailing - Status trailing stop\n"
        "/sync - Sincronizza con Bybit\n"
        "/chiudi SYMBOL - Rimuovi dal tracking\n\n"
        "🎯 <b>Comandi Pattern:</b>\n"
        "/patterns - Lista pattern\n"
        "/pattern_on NOME - Abilita pattern\n"
        "/pattern_off NOME - Disabilita pattern\n"
        "/pattern_info NOME - Info pattern\n\n"
        "📈 <b>Comandi EMA:</b>\n"
        "/ema_filter [MODE] - strict/loose/off\n"
        "/ema_sl [on|off] - EMA Stop Loss\n\n"
        "🔍 <b>Auto-Discovery:</b>\n"
        "/autodiscover [on|off|now|status]\n"
        "→ Top symbols automatici\n\n"
        "📝 <b>Esempi:</b>\n"
        "/analizza BTCUSDT 15m\n"
        "/analizza ETHUSDT 5m yes (con autotrade)\n\n"
        f"⏱️ Timeframes: {', '.join(ENABLED_TFS)}\n"
        f"💰 Rischio default: ${RISK_USD}\n"
        f"🔕 <b>Default:</b> Solo notifiche con pattern\n"
        f"⚠️ <b>NOTA:</b> Solo segnali BUY attivi"
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
    with ACTIVE_ANALYSES_LOCK:
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
    with ACTIVE_ANALYSES_LOCK:
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
        with POSITIONS_LOCK:
            tracked = len(ACTIVE_POSITIONS)
        
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
    
    with POSITIONS_LOCK:
        if symbol in ACTIVE_POSITIONS:
            pos_info = ACTIVE_POSITIONS[symbol]
            del ACTIVE_POSITIONS[symbol]
            
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
        
        with POSITIONS_LOCK:
            tracked_count = len(ACTIVE_POSITIONS)
        
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
        status_emoji = "✅" if AUTO_DISCOVERY_CONFIG['enabled'] else "❌"
        
        msg = f"🔍 <b>Auto-Discovery System</b>\n\n"
        msg += f"Status: {status_emoji} {'Attivo' if AUTO_DISCOVERY_CONFIG['enabled'] else 'Disattivo'}\n\n"
        
        if AUTO_DISCOVERY_CONFIG['enabled']:
            msg += f"<b>Configurazione:</b>\n"
            msg += f"• Top: {AUTO_DISCOVERY_CONFIG['top_count']} symbols\n"
            msg += f"• Timeframe: {AUTO_DISCOVERY_CONFIG['timeframe']}\n"
            msg += f"• Autotrade: {'ON' if AUTO_DISCOVERY_CONFIG['autotrade'] else 'OFF'}\n"
            msg += f"• Update ogni: {AUTO_DISCOVERY_CONFIG['update_interval']//3600}h\n"
            msg += f"• Min volume: ${AUTO_DISCOVERY_CONFIG['min_volume_usdt']/1_000_000:.0f}M\n"
            msg += f"• Min change: +{AUTO_DISCOVERY_CONFIG['min_price_change']}%\n"
            msg += f"• Max change: +{AUTO_DISCOVERY_CONFIG['max_price_change']}%\n\n"
            
            with AUTO_DISCOVERED_LOCK:
                symbols = list(AUTO_DISCOVERED_SYMBOLS)
            
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
        AUTO_DISCOVERY_CONFIG['enabled'] = True
        
        # Schedula il job se non esiste
        current_jobs = context.job_queue.get_jobs_by_name('auto_discovery')
        
        if not current_jobs:
            context.job_queue.run_repeating(
                auto_discover_and_analyze,
                interval=AUTO_DISCOVERY_CONFIG['update_interval'],
                first=60,  # Primo run dopo 1 minuto
                data={'chat_id': chat_id},
                name='auto_discovery'
            )
            
            await update.message.reply_text(
                '✅ <b>Auto-Discovery ATTIVATO</b>\n\n'
                'Primo update tra 1 minuto...\n'
                f"Poi ogni {AUTO_DISCOVERY_CONFIG['update_interval']//3600} ore",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                '✅ <b>Auto-Discovery già attivo</b>',
                parse_mode='HTML'
            )
    
    elif action == 'off':
        AUTO_DISCOVERY_CONFIG['enabled'] = False
        
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
        if not AUTO_DISCOVERY_CONFIG['enabled']:
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
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
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
            
            msg = f"💰 <b>Saldo Wallet ({TRADING_MODE.upper()})</b>\n\n"
            
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
    CON DECIMALI DINAMICI basati sul prezzo
    """
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
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
            
            msg = f"📊 <b>Ultimi {len(pnl_list)} Ordini Chiusi ({TRADING_MODE.upper()})</b>\n\n"
            
            total_pnl = 0
            win_count = 0
            loss_count = 0
            
            for pnl_entry in pnl_list:
                symbol = pnl_entry.get('symbol', 'N/A')
                side = pnl_entry.get('side', 'N/A')
                qty = float(pnl_entry.get('qty', 0))
                avg_entry = float(pnl_entry.get('avgEntryPrice', 0))
                avg_exit = float(pnl_entry.get('avgExitPrice', 0))
                closed_pnl = float(pnl_entry.get('closedPnl', 0))
                
                # Timestamp chiusura (millisecondi)
                updated_time = int(pnl_entry.get('updatedTime', 0))
                close_time = datetime.fromtimestamp(updated_time / 1000, tz=timezone.utc)
                time_str = close_time.strftime('%d/%m %H:%M')
                
                # Statistiche
                total_pnl += closed_pnl
                if closed_pnl > 0:
                    win_count += 1
                else:
                    loss_count += 1
                
                # Emoji in base al risultato
                side_emoji = "🟢" if side == 'Buy' else "🔴"
                pnl_emoji = "✅" if closed_pnl > 0 else "❌"
                
                # Calcola P&L %
                pnl_percent = 0
                if avg_entry > 0:
                    if side == 'Buy':
                        pnl_percent = ((avg_exit - avg_entry) / avg_entry) * 100
                    else:
                        pnl_percent = ((avg_entry - avg_exit) / avg_entry) * 100
                
                # ===== DECIMALI DINAMICI =====
                price_decimals = get_price_decimals(avg_entry)
                
                # Costruisci messaggio ordine
                msg += f"{side_emoji} <b>{symbol}</b> - {side}\n"
                msg += f"  Qty: {qty:.4f}\n"
                msg += f"  Entry: ${avg_entry:.{price_decimals}f}\n"
                msg += f"  Exit: ${avg_exit:.{price_decimals}f}\n"
                msg += f"  {pnl_emoji} PnL: ${closed_pnl:+.2f} ({pnl_percent:+.2f}%)\n"
                msg += f"  Time: {time_str}\n\n"
            
            # Statistiche finali
            msg += "━━━━━━━━━━━━━━━━━━━\n\n"
            msg += f"💰 <b>PnL Totale: ${total_pnl:+.2f}</b>\n"
            msg += f"✅ Win: {win_count} | ❌ Loss: {loss_count}\n"
            
            if (win_count + loss_count) > 0:
                win_rate = (win_count / (win_count + loss_count)) * 100
                msg += f"📊 Win Rate: {win_rate:.1f}%\n"
            
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
    
    with ACTIVE_ANALYSES_LOCK:
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
    
    with ACTIVE_ANALYSES_LOCK:
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
    Comando /trailing
    Mostra status trailing stop loss per tutte le posizioni
    """
    if not TRAILING_STOP_ENABLED:
        await update.message.reply_text(
            '🔕 <b>Trailing Stop Loss DISABILITATO</b>\n\n'
            'Abilita nelle configurazioni: TRAILING_STOP_ENABLED = True',
            parse_mode='HTML'
        )
        return
    
    with POSITIONS_LOCK:
        positions_copy = dict(ACTIVE_POSITIONS)
    
    if not positions_copy:
        await update.message.reply_text(
            '📭 <b>Nessuna posizione attiva</b>\n\n'
            'Non ci sono posizioni con trailing stop loss.',
            parse_mode='HTML'
        )
        return
    
    msg = f"🔄 <b>Trailing Stop Loss Status</b>\n\n"
    msg += f"Check Interval: {TRAILING_CONFIG['check_interval']//60} minuti\n"
    msg += f"Activation: +{TRAILING_CONFIG['activation_percent']}% profit\n\n"
    
    for symbol, pos in positions_copy.items():
        if pos['side'] != 'Buy':
            continue
        
        entry = pos['entry_price']
        current_sl = pos['sl']
        highest = pos.get('highest_price', entry)
        trailing_active = pos.get('trailing_active', False)
        timeframe_entry = pos['timeframe']
        
        # Scarica prezzo corrente
        df = bybit_get_klines(symbol, timeframe_entry, limit=5)
        current_price = df['close'].iloc[-1] if not df.empty else entry
        
        profit_percent = ((current_price - entry) / entry) * 100
        price_decimals = get_price_decimals(current_price)
        
        status_emoji = "✅" if trailing_active else "⏳"
        
        msg += f"{status_emoji} <b>{symbol}</b> ({timeframe_entry})\n"
        msg += f"  Entry: ${entry:.{price_decimals}f}\n"
        msg += f"  Current: ${current_price:.{price_decimals}f}\n"
        msg += f"  Highest: ${highest:.{price_decimals}f}\n"
        msg += f"  SL: ${current_sl:.{price_decimals}f}\n"
        msg += f"  Profit: {profit_percent:+.2f}%\n"
        
        if trailing_active:
            msg += f"  Status: Trailing ATTIVO\n"
        else:
            needed = TRAILING_CONFIG['activation_percent'] - profit_percent
            msg += f"  Status: Serve +{needed:.2f}% per attivare\n"
        
        msg += "\n"
    
    msg += "💡 <b>Info:</b>\n"
    msg += "• SL segue EMA 10 del TF superiore\n"
    msg += "• SL non torna mai indietro\n"
    msg += f"• Buffer: {TRAILING_CONFIG['ema_buffer']*100}% sotto EMA"
    
    await update.message.reply_text(msg, parse_mode='HTML')

async def cmd_test_volume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /test_volume SYMBOL TIMEFRAME
    Testa specificamente il pattern Volume Spike
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /test_volume SYMBOL TIMEFRAME\n'
            'Esempio: /test_volume BTCUSDT 5m'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    
    await update.message.reply_text(f'🔍 Testing Volume Spike pattern su {symbol} {timeframe}...')
    
    try:
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=100)
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato per {symbol}')
            return
        
        # Test filtri globali
        vol_ok = volume_confirmation(df, min_ratio=1.5)
        atr_ok = atr_expanding(df)
        trend_ok = is_uptrend_structure(df)
        
        # Test pattern
        found, data = is_volume_spike_breakout(df)
        
        # Costruisci report
        msg = f"📊 <b>Volume Spike Test: {symbol} {timeframe}</b>\n\n"
        
        msg += "<b>🔍 Filtri Globali:</b>\n"
        msg += f"{'✅' if vol_ok else '❌'} Volume OK (>1.5x media)\n"
        msg += f"{'✅' if atr_ok else '❌'} ATR Expanding\n"
        msg += f"{'✅' if trend_ok else '❌'} Uptrend Structure\n\n"
        
        if found:
            msg += "🎯 <b>PATTERN TROVATO!</b>\n\n"
            msg += f"📈 Volume Ratio: <b>{data['volume_ratio']:.1f}x</b>\n"
            msg += f"💪 Body %: <b>{data['body_pct']*100:.1f}%</b>\n"
            msg += f"📊 EMA 10: ${data['ema10']:.2f}\n"
            msg += f"📊 EMA 60: ${data['ema60']:.2f}\n"
            msg += f"💵 Price: ${data['price']:.2f}\n"
            msg += f"{'✅' if data['breakout_confirmed'] else '⚠️'} Breakout Confermato\n\n"
            msg += "🟢 <b>Pattern VALIDO per entry</b>"
        else:
            msg += "❌ <b>Pattern NON trovato</b>\n\n"
            
            # Debug info
            curr = df.iloc[-1]
            vol = df['volume']
            avg_vol = vol.iloc[-20:-1].mean()
            vol_ratio = vol.iloc[-1] / avg_vol if avg_vol > 0 else 0
            
            msg += "<b>Dettagli candela corrente:</b>\n"
            msg += f"Volume Ratio: {vol_ratio:.1f}x (serve 3x+)\n"
            msg += f"Candela: {'🟢 Verde' if curr['close'] > curr['open'] else '🔴 Rossa'}\n"
            
            body = abs(curr['close'] - curr['open'])
            total_range = curr['high'] - curr['low']
            if total_range > 0:
                body_pct = body / total_range
                msg += f"Body %: {body_pct*100:.1f}% (serve 60%+)\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
    except Exception as e:
        logging.exception('Errore in cmd_test_volume')
        await update.message.reply_text(f'❌ Errore: {str(e)}')


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /list - mostra analisi attive con dettagli completi
    """
    chat_id = update.effective_chat.id
    
    with ACTIVE_ANALYSES_LOCK:
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
        if symbol in ACTIVE_POSITIONS:
            pos = ACTIVE_POSITIONS[symbol]
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

async def cmd_test_compression(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /test_compression SYMBOL TIMEFRAME
    Testa Compression Breakout Enhanced
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /test_compression SYMBOL TIMEFRAME\n'
            'Esempio: /test_compression BTCUSDT 5m'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    
    await update.message.reply_text(f'🔍 Testing Compression Breakout Enhanced su {symbol} {timeframe}...')
    
    try:
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=250)
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato per {symbol}')
            return
        
        # Test filtri globali
        vol_ok = volume_confirmation(df, min_ratio=1.5)
        atr_ok = atr_expanding(df)
        trend_ok = is_uptrend_structure(df)
        
        # Test pattern
        found = is_compression_breakout(df)
        
        # Test HTF resistance
        last_close = df['close'].iloc[-1]
        htf_block = check_compression_htf_resistance(symbol, timeframe, last_close)
        
        # Costruisci report
        msg = f"💥 <b>Compression Breakout Test: {symbol} {timeframe}</b>\n\n"
        
        msg += "<b>🔍 Filtri Globali:</b>\n"
        msg += f"{'✅' if vol_ok else '❌'} Volume OK (>1.5x media)\n"
        msg += f"{'✅' if atr_ok else '❌'} ATR Expanding\n"
        msg += f"{'✅' if trend_ok else '❌'} Uptrend Structure\n\n"
        
        if found:
            msg += "🎯 <b>PATTERN BASE TROVATO!</b>\n\n"
            
            # Calcola metriche enhanced
            vol = df['volume']
            consolidation_vol = vol.iloc[-4:-1].mean()
            breakout_vol = vol.iloc[-2]
            vol_ratio = breakout_vol / consolidation_vol if consolidation_vol > 0 else 0
            
            # RSI
            close = df['close']
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            curr_rsi = rsi.iloc[-1]
            
            # Distance to EMA 10
            ema_10 = df['close'].ewm(span=10, adjust=False).mean()
            curr_ema10 = ema_10.iloc[-1]
            distance = abs(last_close - curr_ema10) / curr_ema10
            
            msg += f"<b>📊 Metriche Enhanced:</b>\n"
            msg += f"Volume Breakout: {vol_ratio:.1f}x {'✅' if vol_ratio >= 1.8 else '❌ (serve 1.8x+)'}\n"
            msg += f"RSI: {curr_rsi:.1f} {'✅' if 50 <= curr_rsi <= 70 else '❌ (serve 50-70)'}\n"
            msg += f"Distance EMA 10: {distance*100:.2f}% {'✅' if distance <= 0.01 else '❌ (max 1%)'}\n\n"
            
            # HTF Check
            msg += f"<b>🔍 HTF Resistance Check:</b>\n"
            if htf_block['blocked']:
                msg += f"❌ BLOCCATO da {htf_block['htf']}\n"
                msg += f"{htf_block['details']}\n\n"
                msg += "🔴 <b>Pattern NON VALIDO (HTF resistance)</b>"
            else:
                msg += f"✅ No resistenza HTF\n\n"
                
                # Verifica TUTTI gli enhancement
                all_checks = (vol_ratio >= 1.8 and 
                            50 <= curr_rsi <= 70 and 
                            distance <= 0.01)
                
                if all_checks:
                    msg += "🟢 <b>Pattern VALIDO (Enhanced)</b>"
                else:
                    msg += "⚠️ <b>Pattern BASE valido MA enhancement falliti</b>"
        else:
            msg += "❌ <b>Pattern BASE NON trovato</b>\n\n"
            msg += "Verifica compressione EMA 5, 10, 223"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
    except Exception as e:
        logging.exception('Errore in cmd_test_compression')
        await update.message.reply_text(f'❌ Errore: {str(e)}')


async def cmd_test_sweep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /test_sweep SYMBOL TIMEFRAME
    Testa specificamente il pattern Liquidity Sweep
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /test_sweep SYMBOL TIMEFRAME\n'
            'Esempio: /test_sweep BTCUSDT 5m'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    
    await update.message.reply_text(f'🔍 Testing Liquidity Sweep su {symbol} {timeframe}...')
    
    try:
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=100)
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato per {symbol}')
            return
        
        # Test pattern
        found, data = is_liquidity_sweep_reversal(df)
        
        # Costruisci report
        msg = f"💎 <b>Liquidity Sweep Test: {symbol} {timeframe}</b>\n\n"
        
        if found:
            msg += "🎯 <b>PATTERN TROVATO!</b>\n\n"
            
            price_decimals = get_price_decimals(data['breakout_price'])
            
            msg += f"📉 <b>Sweep Phase:</b>\n"
            msg += f"  Previous Low: ${data['previous_low']:.{price_decimals}f}\n"
            msg += f"  Sweep Low: ${data['sweep_low']:.{price_decimals}f}\n"
            msg += f"  Distance: {((data['previous_low'] - data['sweep_low']) / data['previous_low'] * 100):.2f}%\n\n"
            
            msg += f"📈 <b>Recovery Phase:</b>\n"
            msg += f"  Recovery: <b>{data['recovery_pct']:.1f}%</b>\n"
            msg += f"  Volume: <b>{data['volume_ratio']:.1f}x</b>\n"
            msg += f"  Recovery High: ${data['recovery_high']:.{price_decimals}f}\n\n"
            
            msg += f"💥 <b>Breakout Phase:</b>\n"
            msg += f"  Breakout Price: ${data['breakout_price']:.{price_decimals}f}\n"
            msg += f"  EMA 10: ${data['ema10']:.{price_decimals}f}\n"
            msg += f"  EMA 60: ${data['ema60']:.{price_decimals}f}\n\n"
            
            msg += f"🎯 <b>Trade Setup:</b>\n"
            msg += f"  Entry: ${data['suggested_entry']:.{price_decimals}f}\n"
            msg += f"  SL: ${data['suggested_sl']:.{price_decimals}f}\n"
            
            # Calcola TP suggerito
            risk = data['suggested_entry'] - data['suggested_sl']
            tp = data['suggested_entry'] + (risk * 2.0)
            msg += f"  TP (2R): ${tp:.{price_decimals}f}\n\n"
            
            msg += "🟢 <b>Pattern VALIDO per entry</b>"
            
        else:
            msg += "❌ <b>Pattern NON trovato</b>\n\n"
            
            # Debug: verifica dove fallisce
            msg += "<b>Checklist:</b>\n"
            
            # Check 1: Previous low
            lookback = 15
            recent_lows = df['low'].iloc[-lookback:-3]
            if len(recent_lows) >= 5:
                previous_low = recent_lows.min()
                touches = (recent_lows <= previous_low * 1.002).sum()
                msg += f"{'✅' if touches >= 2 else '❌'} Previous low valido (touches: {touches})\n"
            else:
                msg += "❌ Dati insufficienti\n"
            
            # Check 2: Sweep
            if len(df) >= 3:
                sweep = df.iloc[-3]
                if len(recent_lows) >= 5:
                    breaks = sweep['low'] < previous_low
                    msg += f"{'✅' if breaks else '❌'} Sweep rompe previous low\n"
            
            # Check 3: Recovery
            if len(df) >= 2:
                recovery = df.iloc[-2]
                is_green = recovery['close'] > recovery['open']
                msg += f"{'✅' if is_green else '❌'} Recovery candle verde\n"
                
                # Volume
                vol = df['volume']
                if len(vol) >= 20:
                    avg_vol = vol.iloc[-20:-2].mean()
                    recovery_vol = vol.iloc[-2]
                    if avg_vol > 0:
                        vol_ratio = recovery_vol / avg_vol
                        msg += f"{'✅' if vol_ratio >= 2.0 else '❌'} Volume recovery: {vol_ratio:.1f}x (serve 2x+)\n"
            
            # Check 4: Breakout
            if len(df) >= 2:
                current = df.iloc[-1]
                recovery = df.iloc[-2]
                breaks = current['close'] > recovery['high']
                msg += f"{'✅' if breaks else '❌'} Breakout recovery high\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
    except Exception as e:
        logging.exception('Errore in cmd_test_sweep')
        await update.message.reply_text(f'❌ Errore: {str(e)}')


async def cmd_test_sr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /test_sr SYMBOL TIMEFRAME
    Testa specificamente il pattern S/R Bounce
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /test_sr SYMBOL TIMEFRAME\n'
            'Esempio: /test_sr BTCUSDT 5m'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    
    await update.message.reply_text(f'🔍 Testing S/R Bounce su {symbol} {timeframe}...')
    
    try:
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=100)
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato per {symbol}')
            return
        
        # Test filtri globali
        vol_ok = volume_confirmation(df, min_ratio=1.5)
        atr_ok = atr_expanding(df)
        trend_ok = is_uptrend_structure(df)
        
        # Test pattern
        found, data = is_support_resistance_bounce(df)
        
        # Costruisci report
        msg = f"🎯 <b>S/R Bounce Test: {symbol} {timeframe}</b>\n\n"
        
        msg += "<b>🔍 Filtri Globali:</b>\n"
        msg += f"{'✅' if vol_ok else '❌'} Volume OK (>1.5x media)\n"
        msg += f"{'✅' if atr_ok else '❌'} ATR Expanding\n"
        msg += f"{'✅' if trend_ok else '❌'} Uptrend Structure\n\n"
        
        if found:
            price_decimals = get_price_decimals(data['support_level'])
            
            msg += "🎯 <b>PATTERN TROVATO!</b>\n\n"
            
            msg += f"📊 <b>Support Level:</b>\n"
            msg += f"  Level: ${data['support_level']:.{price_decimals}f}\n"
            msg += f"  Touches: <b>{data['touches']}</b> volte\n"
            msg += f"  Distance: ${data['distance_to_support']:.{price_decimals}f}\n\n"
            
            msg += f"🕯️ <b>Candela:</b>\n"
            msg += f"  Body: {data['body_pct']*100:.1f}% del range\n"
            msg += f"  Lower Wick: {data['lower_wick_pct']*100:.1f}% del range\n"
            msg += f"  Rejection: <b>{data['rejection_strength']:.2f}x</b> corpo\n\n"
            
            msg += f"📈 <b>Volume & EMA:</b>\n"
            msg += f"  Volume: <b>{data['volume_ratio']:.1f}x</b> media\n"
            msg += f"  EMA 10: ${data['ema10']:.{price_decimals}f}\n"
            msg += f"  EMA 60: ${data['ema60']:.{price_decimals}f}\n"
            msg += f"  Near EMA 60: {'✅ Yes' if data['near_ema60'] else '⚠️ No'}\n\n"
            
            # Calcola setup
            curr_price = df['close'].iloc[-1]
            sl = data['support_level'] * 0.998
            risk = curr_price - sl
            tp = curr_price + (risk * 1.6)
            
            msg += f"🎯 <b>Trade Setup:</b>\n"
            msg += f"  Entry: ${curr_price:.{price_decimals}f}\n"
            msg += f"  SL: ${sl:.{price_decimals}f}\n"
            msg += f"  TP: ${tp:.{price_decimals}f} (1.6R)\n\n"
            
            msg += "🟢 <b>Pattern VALIDO per entry</b>"
            
        else:
            msg += "❌ <b>Pattern NON trovato</b>\n\n"
            
            # Debug
            curr = df.iloc[-1]
            lookback_lows = df['low'].iloc[-50:-1]
            sorted_lows = lookback_lows.nsmallest(5)
            support_level = sorted_lows.mean()
            
            msg += "<b>Checklist:</b>\n"
            
            # Check 1: Support valido
            tolerance = support_level * 0.005
            touches = (lookback_lows <= support_level + tolerance).sum()
            msg += f"{'✅' if touches >= 3 else '❌'} Support valido ({touches} touches, serve 3+)\n"
            
            # Check 2: Tocca support
            touches_support = abs(curr['low'] - support_level) <= tolerance
            distance_pct = abs(curr['low'] - support_level) / support_level * 100
            msg += f"{'✅' if touches_support else '❌'} Tocca support (dist: {distance_pct:.2f}%, max 0.5%)\n"
            
            # Check 3: Candela verde
            is_green = curr['close'] > curr['open']
            msg += f"{'✅' if is_green else '❌'} Candela verde\n"
            
            # Check 4: Rejection
            if is_green:
                lower_wick = min(curr['open'], curr['close']) - curr['low']
                body = abs(curr['close'] - curr['open'])
                has_rejection = lower_wick >= body if body > 0 else False
                msg += f"{'✅' if has_rejection else '❌'} Rejection (wick >= corpo)\n"
            
            # Check 5: Volume
            vol = df['volume']
            if len(vol) >= 20:
                avg_vol = vol.iloc[-20:-1].mean()
                curr_vol = vol.iloc[-1]
                vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 0
                msg += f"{'✅' if vol_ratio >= 1.2 else '❌'} Volume: {vol_ratio:.1f}x (serve 1.2x+)\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
    except Exception as e:
        logging.exception('Errore in cmd_test_sr')
        await update.message.reply_text(f'❌ Errore: {str(e)}')


async def cmd_debug_volume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /debug_volume SYMBOL TIMEFRAME
    Mostra dettagli completi sul volume per debugging
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /debug_volume SYMBOL TIMEFRAME\n'
            'Esempio: /debug_volume BTCUSDT 5m'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    
    await update.message.reply_text(f'🔍 Debug volume per {symbol} {timeframe}...')
    
    try:
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=50)
        
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato per {symbol}')
            return
        
        msg = f"🔍 <b>Debug Volume: {symbol} {timeframe}</b>\n\n"
        
        # Check colonne
        msg += f"<b>Colonne DataFrame:</b>\n"
        msg += f"{df.columns.tolist()}\n\n"
        
        # Check volume column
        if 'volume' in df.columns:
            msg += f"<b>✅ Volume column EXISTS</b>\n\n"
            
            vol = df['volume']
            
            # Stats
            msg += f"<b>Volume Stats:</b>\n"
            msg += f"Count: {len(vol)}\n"
            msg += f"Sum: {vol.sum():.2f}\n"
            msg += f"Mean: {vol.mean():.2f}\n"
            msg += f"Min: {vol.min():.2f}\n"
            msg += f"Max: {vol.max():.2f}\n"
            msg += f"Current: {vol.iloc[-1]:.2f}\n\n"
            
            # Check NaN
            nan_count = vol.isna().sum()
            msg += f"NaN values: {nan_count}\n"
            msg += f"Zero values: {(vol == 0).sum()}\n\n"
            
            # Volume ratio
            if len(vol) >= 20:
                avg_vol = vol.iloc[-20:-1].mean()
                current_vol = vol.iloc[-1]
                
                if avg_vol > 0:
                    ratio = current_vol / avg_vol
                    msg += f"<b>Volume Ratio Check:</b>\n"
                    msg += f"Avg (20 periods): {avg_vol:.2f}\n"
                    msg += f"Current: {current_vol:.2f}\n"
                    msg += f"Ratio: {ratio:.2f}x\n"
                    msg += f"Threshold: 1.5x\n"
                    msg += f"Result: {'✅ PASS' if ratio > 1.5 else '❌ FAIL'}\n\n"
                else:
                    msg += f"❌ Average volume is ZERO!\n\n"
            
            # Ultimi 10 volumi
            msg += f"<b>Last 10 volumes:</b>\n"
            last_10 = vol.iloc[-10:].tolist()
            for i, v in enumerate(last_10, 1):
                msg += f"{i}. {v:.2f}\n"
        else:
            msg += f"❌ <b>Volume column NOT FOUND!</b>\n"
            msg += f"Available: {df.columns.tolist()}"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
    except Exception as e:
        logging.exception('Error in cmd_debug_volume')
        await update.message.reply_text(f'❌ Errore: {str(e)}')


async def cmd_test_flag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /test_flag SYMBOL TIMEFRAME
    Testa Bullish Flag Breakout Enhanced
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /test_flag SYMBOL TIMEFRAME\n'
            'Esempio: /test_flag BTCUSDT 5m'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    
    await update.message.reply_text(f'🔍 Testing Bullish Flag su {symbol} {timeframe}...')
    
    try:
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=100)
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato per {symbol}')
            return
        
        # Test filtri globali
        vol_ok = volume_confirmation(df, min_ratio=1.5)
        atr_ok = atr_expanding(df)
        trend_ok = is_uptrend_structure(df)
        
        # Test pattern
        found, data = is_bullish_flag_breakout(df)
        
        # Costruisci report
        msg = f"🚩 <b>Bullish Flag Test: {symbol} {timeframe}</b>\n\n"
        
        msg += "<b>🔍 Filtri Globali:</b>\n"
        msg += f"{'✅' if vol_ok else '❌'} Volume OK (>1.5x media)\n"
        msg += f"{'✅' if atr_ok else '❌'} ATR Expanding\n"
        msg += f"{'✅' if trend_ok else '❌'} Uptrend Structure\n\n"
        
        if found:
            price_decimals = get_price_decimals(data['X'])
            
            msg += "🎯 <b>PATTERN TROVATO!</b>\n\n"
            
            msg += f"<b>📊 Pole (Candela Iniziale):</b>\n"
            msg += f"  Height: <b>{data['pole_height_pct']:.2f}%</b>\n"
            msg += f"  Body: {data['pole_body_pct']:.1f}% del range\n"
            msg += f"  Valido: {'✅' if data['pole_height_pct'] >= 0.8 else '❌'} (min 0.8%)\n\n"
            
            msg += f"<b>🏴 Flag (Consolidamento):</b>\n"
            msg += f"  Duration: <b>{data['flag_duration']}</b> candele\n"
            msg += f"  Valido: {'✅' if 3 <= data['flag_duration'] <= 8 else '❌'} (range 3-8)\n"
            msg += f"  Low: ${data['consolidation_low']:.{price_decimals}f}\n\n"
            
            msg += f"<b>💥 Breakout:</b>\n"
            msg += f"  X (breakout level): ${data['X']:.{price_decimals}f}\n"
            msg += f"  Current Price: ${data['current_price']:.{price_decimals}f}\n"
            msg += f"  Volume: <b>{data['volume_ratio']:.1f}x</b> consolidamento\n"
            msg += f"  Valido: {'✅' if data['volume_ratio'] >= 2.0 else '❌'} (min 2x)\n\n"
            
            # Calcola setup
            entry = data['X']
            sl = data['consolidation_low'] * 0.998
            tp = data['X'] + (data['pole_height'] * 1.5)
            
            msg += f"<b>🎯 Trade Setup:</b>\n"
            msg += f"  Entry: ${entry:.{price_decimals}f}\n"
            msg += f"  SL: ${sl:.{price_decimals}f}\n"
            msg += f"  TP: ${tp:.{price_decimals}f} (1.5x pole)\n\n"
            
            # Validation summary
            all_checks = (data['pole_height_pct'] >= 0.8 and
                         3 <= data['flag_duration'] <= 8 and
                         data['volume_ratio'] >= 2.0)
            
            if all_checks:
                msg += "🟢 <b>Pattern VALIDO (Enhanced)</b>"
            else:
                msg += "⚠️ <b>Pattern trovato MA non passa tutti i check</b>\n"
                if data['pole_height_pct'] < 0.8:
                    msg += "• Pole troppo piccolo (<0.8%)\n"
                if data['flag_duration'] < 3 or data['flag_duration'] > 8:
                    msg += "• Flag duration fuori range (3-8)\n"
                if data['volume_ratio'] < 2.0:
                    msg += "• Volume insufficiente (<2x)\n"
            
        else:
            msg += "❌ <b>Pattern NON trovato</b>\n\n"
            
            # Debug info
            msg += "<b>Possibili motivi:</b>\n"
            msg += "• Nessun pole valido (corpo minore 60%, height minore 0.8%)\n"
            msg += "• Flag troppo corto (minore 3) o lungo (maggiore 8)\n"
            msg += "• Candele flag superano X\n"
            msg += "• Volume breakout minore 2x consolidamento\n"
            msg += "• Nessun breakout confermato (close minore X)\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
    except Exception as e:
        logging.exception('Errore in cmd_test_flag')
        await update.message.reply_text(f'❌ Errore: {str(e)}')


async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /test SYMBOL TIMEFRAME
    Testa i pattern sull'ultima candela e mostra debug info
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /test SYMBOL TIMEFRAME\n'
            'Esempio: /test BTCUSDT 15m\n'
            'Questo comando mostra info dettagliate sui pattern rilevati'
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
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=200)
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato per {symbol}')
            return
        
        # Analizza pattern
        found, side, pattern, pattern_data = check_patterns(df)
        
        # Info candele recenti
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Calcola metriche
        last_body = abs(last['close'] - last['open'])
        last_range = last['high'] - last['low']
        last_body_pct = (last_body / last_range * 100) if last_range > 0 else 0
        
        lower_wick = min(last['open'], last['close']) - last['low']
        upper_wick = last['high'] - max(last['open'], last['close'])
        
        # Test individuali
        tests = {
            '🚩 Bullish Flag Breakout': is_bullish_flag_breakout(df)[0],  # Ritorna (bool, data)
            '⭐ Morning Star': morning_star_ema_breakout(df),
            '💥 Compression Breakout': is_compression_breakout(df),
            '🔄 Bullish Comeback': is_bullish_comeback(df),
            '🟢 Bullish Engulfing': is_bullish_engulfing(prev, last),
            '🔴 Bearish Engulfing': is_bearish_engulfing(prev, last),
            '🔨 Hammer': is_hammer(last),
            '💫 Shooting Star': is_shooting_star(last),
            '📍 Pin Bar': is_pin_bar(last),
            '➖ Doji': is_doji(last),
            '⭐ Morning Star': is_morning_star(prev2, prev, last),
            '🌙 Evening Star': is_evening_star(prev2, prev, last),
            '⬆️ Three White Soldiers': is_three_white_soldiers(prev2, prev, last),
            '⬇️ Three Black Crows': is_three_black_crows(prev2, prev, last)
        }
        
        # Costruisci messaggio
        msg = f"🔍 <b>Test Pattern: {symbol} {timeframe}</b>\n\n"
        
        if found:
            msg += f"✅ <b>PATTERN TROVATO: {pattern}</b>\n"
            msg += f"📈 Direzione: {side}\n\n"
        else:
            msg += "❌ Nessun pattern rilevato\n\n"
        
        msg += f"📊 <b>Ultima candela:</b>\n"
        msg += f"O: ${last['open']:.2f} | H: ${last['high']:.2f}\n"
        msg += f"L: ${last['low']:.2f} | C: ${last['close']:.2f}\n"
        msg += f"{'🟢 Bullish' if last['close'] > last['open'] else '🔴 Bearish'}\n"
        msg += f"Corpo: {last_body_pct:.1f}% del range\n"
        msg += f"Ombra inf: ${lower_wick:.2f} ({lower_wick/last_range*100:.1f}%)\n"
        msg += f"Ombra sup: ${upper_wick:.2f} ({upper_wick/last_range*100:.1f}%)\n\n"
        
        msg += "🧪 <b>Test Pattern:</b>\n"
        for pattern_name, result in tests.items():
            emoji = "✅" if result else "❌"
            msg += f"{emoji} {pattern_name}\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
        # Invia anche il grafico
        try:
            chart_buffer = generate_chart(df, symbol, timeframe)
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=chart_buffer,
                caption=f"Grafico di test per {symbol} {timeframe}"
            )
        except Exception as e:
            logging.error(f'Errore generazione grafico test: {e}')
    
    except Exception as e:
        logging.exception('Errore in cmd_test')
        await update.message.reply_text(f'❌ Errore: {str(e)}')


async def cmd_test_breakout_retest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /test_br SYMBOL TIMEFRAME
    Testa Breakout + Retest pattern
    """
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /test_br SYMBOL TIMEFRAME\n'
            'Esempio: /test_br BTCUSDT 5m'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    
    await update.message.reply_text(
        f'🔍 Testing Breakout + Retest su {symbol} {timeframe}...'
    )
    
    try:
        df = bybit_get_klines(symbol, timeframe, limit=50)
        
        if df.empty:
            await update.message.reply_text(f'❌ Nessun dato per {symbol}')
            return
        
        # Test pattern
        found, data = is_breakout_retest(df)
        
        msg = f"🔄 <b>Breakout + Retest Test: {symbol} {timeframe}</b>\n\n"
        
        if found:
            price_decimals = get_price_decimals(data['resistance'])
            
            msg += "🎯 <b>PATTERN TROVATO!</b>\n\n"
            
            msg += f"<b>📊 Consolidamento:</b>\n"
            msg += f"  Range: {data['range_pct']:.2f}%\n"
            msg += f"  Resistance: ${data['resistance']:.{price_decimals}f}\n"
            msg += f"  Support: ${data['support']:.{price_decimals}f}\n"
            msg += f"  Touches R/S: {data['touches_resistance']}/{data['touches_support']}\n\n"
            
            msg += f"<b>💥 Breakout:</b>\n"
            msg += f"  Price: ${data['breakout_price']:.{price_decimals}f}\n"
            msg += f"  Volume: {data['breakout_vol_ratio']:.1f}x\n"
            msg += f"  Body: {data['breakout_body_pct']:.1f}%\n\n"
            
            msg += f"<b>🔄 Retest:</b>\n"
            msg += f"  Low: ${data['retest_low']:.{price_decimals}f}\n"
            msg += f"  Distance to R: ${data['distance_to_resistance']:.{price_decimals}f}\n"
            msg += f"  Rejection: {data['retest_rejection_pct']:.1f}%\n"
            msg += f"  Volume: {data['retest_vol_ratio']:.1f}x\n"
            msg += f"  Pullback: {data['pullback_duration']} candele\n\n"
            
            msg += f"<b>🎯 Trade Setup:</b>\n"
            msg += f"  Entry: ${data['suggested_entry']:.{price_decimals}f}\n"
            msg += f"  SL: ${data['suggested_sl']:.{price_decimals}f}\n"
            msg += f"  TP: ${data['suggested_tp']:.{price_decimals}f}\n\n"
            
            # Calcola R:R
            risk = data['suggested_entry'] - data['suggested_sl']
            reward = data['suggested_tp'] - data['suggested_entry']
            rr = reward / risk if risk > 0 else 0
            
            msg += f"📏 R:R: {rr:.2f}:1\n\n"
            msg += "🟢 <b>Pattern VALIDO per entry</b>"
            
        else:
            msg += "❌ <b>Pattern NON trovato</b>\n\n"
            msg += "<b>Possibili motivi:</b>\n"
            msg += "• Nessun consolidamento valido (5-10 candele)\n"
            msg += "• Breakout debole (volume <2x, corpo <60%)\n"
            msg += "• Pullback invalido (rompe resistance)\n"
            msg += "• Retest assente o troppo lontano\n"
            msg += "• No rejection sul retest (wick <40%)\n"
            msg += "• Prezzo sotto EMA 10 o 60\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
    except Exception as e:
        logging.exception('Errore in cmd_test_breakout_retest')
        await update.message.reply_text(f'❌ Errore: {str(e)}')


# ----------------------------- MAIN -----------------------------

def main():
    """Funzione principale"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,  # 👈 Cambia da INFO a DEBUG per vedere i filtri
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

        # Avvia Auto-Discovery se abilitato
    if AUTO_DISCOVERY_ENABLED and AUTO_DISCOVERY_CONFIG['enabled']:
        # Nota: Serve chat_id, quindi auto-discovery sarà attivato
        # dal primo utente che usa /autodiscover on
        logging.info('🔍 Auto-Discovery configurato (attiva con /autodiscover on)')
    
    # Verifica variabili d'ambiente
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == '':
        logging.error('❌ TELEGRAM_TOKEN non configurato!')
        logging.error('Imposta la variabile d\'ambiente TELEGRAM_TOKEN')
        return
    
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        logging.warning('⚠️ Bybit API keys non configurate. Trading disabilitato.')
    
    # Crea applicazione con JobQueue
    try:
        from telegram.ext import JobQueue
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
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
    application.add_handler(CommandHandler('ema_sl', cmd_ema_sl))
    application.add_handler(CommandHandler('test_volume', cmd_test_volume))
    application.add_handler(CommandHandler('test_sweep', cmd_test_sweep))
    application.add_handler(CommandHandler('test_sr', cmd_test_sr))
    application.add_handler(CommandHandler('test_compression', cmd_test_compression))
    application.add_handler(CommandHandler('debug_volume', cmd_debug_volume))
    application.add_handler(CommandHandler('test_flag', cmd_test_flag))
    application.add_handler(CommandHandler('test_br', cmd_test_breakout_retest))

    # Schedula trailing stop loss job
    schedule_trailing_stop_job(application)
    
    # Avvia bot
    mode_emoji = "🎮" if TRADING_MODE == 'demo' else "⚠️💰"
    logging.info('🚀 Bot avviato correttamente!')
    logging.info(f'{mode_emoji} Modalità Trading: {TRADING_MODE.upper()}')
    logging.info(f'⏱️ Timeframes supportati: {ENABLED_TFS}')
    logging.info(f'💰 Rischio per trade: ${RISK_USD}')
    
    if TRADING_MODE == 'live':
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

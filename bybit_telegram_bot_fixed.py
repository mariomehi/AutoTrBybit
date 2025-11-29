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
            # MUST: Prezzo sopra EMA 10
            'price_above_ema10': True,
            # BONUS: EMA 5 sopra EMA 10 (momentum forte)
            'ema5_above_ema10': True,
            # GOLD: Pattern vicino a EMA 10 (pullback)
            'near_ema10': False  # Opzionale per scalping
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

# Pattern Management System
AVAILABLE_PATTERNS = {
    'bullish_comeback': {
        'name': 'Bullish Comeback',
        'enabled': True,
        'description': 'Inversione/rigetto rialzista (2 varianti)',
        'side': 'Buy',
        'emoji': '🔄'
    },
    'compression_breakout': { 
        'name': 'Compression Breakout',
        'enabled': True,
        'description': 'Breakout esplosivo dopo compressione EMA',
        'side': 'Buy',
        'emoji': '💥'
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
        'enabled': True,
        'description': '3 candele: ribassista, piccola, rialzista',
        'side': 'Buy',
        'emoji': '⭐'
    },
    'three_white_soldiers': {
        'name': 'Three White Soldiers',
        'enabled': True,
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


def analyze_ema_conditions(df: pd.DataFrame, timeframe: str):
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
    
    # === SCALPING RULES (5m, 15m) ===
    if timeframe in ['5m', '15m']:
        # MUST: Prezzo sopra EMA 10
        if rules.get('price_above_ema10'):
            if last_close > last_ema10:
                conditions['price_above_ema10'] = True
                score += 40
                details.append("✅ Prezzo > EMA 10")
            else:
                conditions['price_above_ema10'] = False
                score -= 30
                details.append("❌ Prezzo < EMA 10")
        
        # BONUS: EMA 5 sopra EMA 10
        if rules.get('ema5_above_ema10'):
            if last_ema5 > last_ema10:
                conditions['ema5_above_ema10'] = True
                score += 30
                details.append("✅ EMA 5 > EMA 10 (momentum)")
            else:
                conditions['ema5_above_ema10'] = False
                score += 10  # Non critico
                details.append("⚠️ EMA 5 < EMA 10")
        
        # GOLD: Pattern vicino a EMA 10 (pullback)
        distance_to_ema10 = abs(last_close - last_ema10) / last_ema10
        if distance_to_ema10 < 0.005:  # Entro 0.5%
            conditions['near_ema10'] = True
            score += 30
            details.append("🌟 Vicino EMA 10 (pullback zone!)")
        else:
            conditions['near_ema10'] = False
    
    # === DAY TRADING RULES (30m, 1h) ===
    elif timeframe in ['30m', '1h']:
        # MUST: Prezzo sopra EMA 60
        if rules.get('price_above_ema60'):
            if last_close > last_ema60:
                conditions['price_above_ema60'] = True
                score += 40
                details.append("✅ Prezzo > EMA 60")
            else:
                conditions['price_above_ema60'] = False
                score -= 30
                details.append("❌ Prezzo < EMA 60")
        
        # BONUS: EMA 10 sopra EMA 60
        if rules.get('ema10_above_ema60'):
            if last_ema10 > last_ema60:
                conditions['ema10_above_ema60'] = True
                score += 30
                details.append("✅ EMA 10 > EMA 60 (trend ok)")
            else:
                conditions['ema10_above_ema60'] = False
                score += 10
                details.append("⚠️ EMA 10 < EMA 60")
        
        # GOLD: Pattern vicino a EMA 60
        distance_to_ema60 = abs(last_close - last_ema60) / last_ema60
        if distance_to_ema60 < 0.01:  # Entro 1%
            conditions['near_ema60'] = True
            score += 30
            details.append("🌟 Vicino EMA 60 (bounce zone!)")
        else:
            conditions['near_ema60'] = False
    
    # === SWING TRADING RULES (4h) ===
    elif timeframe in ['4h']:
        # MUST: Prezzo sopra EMA 223
        if rules.get('price_above_ema223'):
            if last_close > last_ema223:
                conditions['price_above_ema223'] = True
                score += 40
                details.append("✅ Prezzo > EMA 223 (bull market)")
            else:
                conditions['price_above_ema223'] = False
                score -= 30
                details.append("❌ Prezzo < EMA 223 (bear market)")
        
        # BONUS: EMA 60 sopra EMA 223
        if rules.get('ema60_above_ema223'):
            if last_ema60 > last_ema223:
                conditions['ema60_above_ema223'] = True
                score += 30
                details.append("✅ EMA 60 > EMA 223 (strong trend)")
            else:
                conditions['ema60_above_ema223'] = False
                score += 10
                details.append("⚠️ EMA 60 < EMA 223")
        
        # GOLD: Pattern vicino a EMA 223
        distance_to_ema223 = abs(last_close - last_ema223) / last_ema223
        if distance_to_ema223 < 0.02:  # Entro 2%
            conditions['near_ema223'] = True
            score += 30
            details.append("🌟 Vicino EMA 223 (major support!)")
        else:
            conditions['near_ema223'] = False
    
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


# ----------------------------- PATTERN DETECTION -----------------------------

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
    Pattern: Compression Breakout
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
    """
    if len(df) < 50:  # Servono dati per EMA 223
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
    
    # Check volume (opzionale - solo se disponibile)
    volume_ok = True
    if 'volume' in df.columns:
        vol = df['volume']
        if len(vol) >= 21:
            avg_vol = vol.iloc[-21:-1].mean()
            prev_vol = vol.iloc[-2]
            # Volume breakout deve essere superiore alla media
            volume_ok = prev_vol > avg_vol * 1.2
    
    has_confirmation = (curr_not_bearish and 
                       stays_above and 
                       ema5_above_10 and 
                       no_deep_retest and
                       volume_ok)
    
    return has_confirmation


def check_patterns(df: pd.DataFrame):
    """
    Controlla tutti i pattern ABILITATI
    Returns: (found: bool, side: str, pattern_name: str)
    """
    if len(df) < 6:
        return (False, None, None)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # ===== PATTERN BUY =====
    
    # Bullish Comeback (controlla per primo perché più specifico)
    if AVAILABLE_PATTERNS['bullish_comeback']['enabled']:
        if is_bullish_comeback(df):
            return (True, 'Buy', 'Bullish Comeback')

    # Compression Breakout (molto specifico, alta priorità)
    if AVAILABLE_PATTERNS['compression_breakout']['enabled']:
        if is_compression_breakout(df):
            return (True, 'Buy', 'Compression Breakout')
    
    # Bullish Engulfing
    if AVAILABLE_PATTERNS['bullish_engulfing']['enabled']:
        if is_bullish_engulfing(prev, last):
            return (True, 'Buy', 'Bullish Engulfing')
    
    # Bearish Engulfing (SELL)
    if AVAILABLE_PATTERNS['bearish_engulfing']['enabled']:
        if is_bearish_engulfing(prev, last):
            return (True, 'Sell', 'Bearish Engulfing')
    
    # Hammer
    if AVAILABLE_PATTERNS['hammer']['enabled']:
        if is_hammer(last):
            return (True, 'Buy', 'Hammer')
    
    # Shooting Star (SELL)
    if AVAILABLE_PATTERNS['shooting_star']['enabled']:
        if is_shooting_star(last):
            return (True, 'Sell', 'Shooting Star')
    
    # Pin bar
    if AVAILABLE_PATTERNS['pin_bar_bullish']['enabled'] or AVAILABLE_PATTERNS['pin_bar_bearish']['enabled']:
        if is_pin_bar(last):
            lower_wick = min(last['open'], last['close']) - last['low']
            upper_wick = last['high'] - max(last['open'], last['close'])
            
            if lower_wick > upper_wick:
                # Pin bar bullish
                if AVAILABLE_PATTERNS['pin_bar_bullish']['enabled']:
                    return (True, 'Buy', 'Pin Bar Bullish')
            else:
                # Pin bar bearish
                if AVAILABLE_PATTERNS['pin_bar_bearish']['enabled']:
                    return (True, 'Sell', 'Pin Bar Bearish')
    
    # Doji
    if AVAILABLE_PATTERNS['doji']['enabled']:
        if is_doji(last):
            # Il doji può essere sia BUY che SELL a seconda del trend
            if prev['close'] > prev['open']:
                return (True, 'Sell', 'Doji (reversione)')
            else:
                return (True, 'Buy', 'Doji (reversione)')
    
    # Morning Star
    if AVAILABLE_PATTERNS['morning_star']['enabled']:
        if is_morning_star(prev2, prev, last):
            return (True, 'Buy', 'Morning Star')
    
    # Evening Star (SELL)
    if AVAILABLE_PATTERNS['evening_star']['enabled']:
        if is_evening_star(prev2, prev, last):
            return (True, 'Sell', 'Evening Star')
    
    # Three White Soldiers
    if AVAILABLE_PATTERNS['three_white_soldiers']['enabled']:
        if is_three_white_soldiers(prev2, prev, last):
            return (True, 'Buy', 'Three White Soldiers')
    
    # Three Black Crows (SELL)
    if AVAILABLE_PATTERNS['three_black_crows']['enabled']:
        if is_three_black_crows(prev2, prev, last):
            return (True, 'Sell', 'Three Black Crows')
    
    return (False, None, None)


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


async def place_bybit_order(symbol: str, side: str, qty: float, sl_price: float, tp_price: float):
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
        
        # Arrotonda prezzi (2 decimali per la maggior parte dei symbol)
        # Per symbol con prezzi molto bassi (<1), usa 4 decimali
        price_decimals = 4 if sl_price < 1 else 2
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
                    'sl': sl_price,
                    'tp': tp_price,
                    'order_id': order.get('result', {}).get('orderId'),
                    'timestamp': datetime.now(timezone.utc).isoformat()
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
    
    COMPORTAMENTO DEFAULT: Invia grafico SOLO quando trova un pattern
    Se symbol-timeframe è in FULL_NOTIFICATIONS: Invia SEMPRE (anche senza pattern)
    """
    job_ctx = context.job.data
    chat_id = job_ctx['chat_id']
    symbol = job_ctx['symbol']
    timeframe = job_ctx['timeframe']
    key = f'{symbol}-{timeframe}'

    # Verifica se le notifiche complete sono attive per questo symbol/timeframe
    with FULL_NOTIFICATIONS_LOCK:
        full_mode = chat_id in FULL_NOTIFICATIONS and key in FULL_NOTIFICATIONS[chat_id]

    try:
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=200)
        if df.empty:
            logging.warning(f'Nessun dato per {symbol} {timeframe}')
            # Invia errore solo se full mode attivo
            if full_mode:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f'⚠️ Nessun dato disponibile per {symbol} {timeframe}'
                )
            return

        last_close = df['close'].iloc[-1]
        last_time = df.index[-1]
        
        # Controlla pattern
        found, side, pattern = check_patterns(df)
        
        # Se NON c'è pattern e NON siamo in full mode, skip completamente
        if not found and not full_mode:
            logging.debug(f'🔕 {symbol} {timeframe} - nessun pattern, skip notifica (default mode)')
            return
        
        # === ANALISI EMA (solo per pattern BUY) ===
        ema_analysis = None
        if found and side == 'Buy' and EMA_FILTER_ENABLED:
            ema_analysis = analyze_ema_conditions(df, timeframe)
            
            # Se modalità STRICT e non passa il filtro, skip
            if EMA_FILTER_MODE == 'strict' and not ema_analysis['passed']:
                logging.info(f'⚠️ Pattern {pattern} su {symbol} {timeframe} FILTRATO da EMA (score: {ema_analysis["score"]})')
                # Non inviare segnale in modalità strict
                if not full_mode:
                    return
        
        # Calcola ATR per eventuali SL/TP
        atr_series = atr(df, period=14)
        last_atr = atr_series.iloc[-1] if not atr_series.isna().all() else np.nan
        
        # Prepara messaggio base
        timestamp_str = last_time.strftime('%Y-%m-%d %H:%M UTC')
        caption = (
            f"📊 <b>{symbol}</b> ({timeframe})\n"
            f"🕐 {timestamp_str}\n"
            f"💵 Prezzo: ${last_close:.4f}\n"
        )
        
        # Se c'è volume, mostralo
        if VOLUME_FILTER:
            vol = df['volume']
            if len(vol) >= 21:
                avg_vol = vol.iloc[-21:-1].mean()
                current_vol = vol.iloc[-1]
                vol_ratio = (current_vol / avg_vol) if avg_vol > 0 else 0
                caption += f"📈 Volume: {vol_ratio:.2f}x media\n"
        
        # Se pattern trovato, aggiungi dettagli
        if found:
            logging.info(f'🎯 SEGNALE: {pattern} - {side} su {symbol} {timeframe}')
            
            # Calcola SL basato su EMA o ATR
            if USE_EMA_STOP_LOSS:
                sl_price, ema_used, ema_value = calculate_ema_stop_loss(df, timeframe, last_close, side)
            else:
                # Fallback ATR tradizionale
                if not math.isnan(last_atr) and last_atr > 0:
                    if side == 'Buy':
                        sl_price = last_close - last_atr * ATR_MULT_SL
                    else:
                        sl_price = last_close + last_atr * ATR_MULT_SL
                    ema_used = 'ATR'
                    ema_value = last_atr
                else:
                    if side == 'Buy':
                        sl_price = df['low'].iloc[-1]
                    else:
                        sl_price = df['high'].iloc[-1]
                    ema_used = 'Candle Low/High'
                    ema_value = 0
            
            # Calcola TP (sempre con ATR)
            if not math.isnan(last_atr) and last_atr > 0:
                if side == 'Buy':
                    tp_price = last_close + last_atr * ATR_MULT_TP
                else:
                    tp_price = last_close - last_atr * ATR_MULT_TP
            else:
                if side == 'Buy':
                    tp_price = last_close * 1.02
                else:
                    tp_price = last_close * 0.98
            
            # Calcola position size
            qty = calculate_position_size(last_close, sl_price, RISK_USD)
            
            # Usa risk override se disponibile per questo symbol
            risk_for_symbol = SYMBOL_RISK_OVERRIDE.get(symbol, RISK_USD)
            if risk_for_symbol != RISK_USD:
                qty = calculate_position_size(last_close, sl_price, risk_for_symbol)
                logging.info(f'💰 Using risk override for {symbol}: ${risk_for_symbol}')
            
            # Verifica se esiste già una posizione
            position_exists = symbol in ACTIVE_POSITIONS
            
            # Costruisci messaggio segnale
            quality_emoji = {
                'GOLD': '🌟',
                'GOOD': '✅',
                'OK': '⚠️',
                'WEAK': '🔶',
                'BAD': '❌'
            }
            
            caption = f"🔥 <b>SEGNALE TROVATO!</b>\n\n"
            
            # Se c'è analisi EMA, mostra quality
            if ema_analysis:
                q_emoji = quality_emoji.get(ema_analysis['quality'], '⚪')
                caption += f"{q_emoji} <b>Quality: {ema_analysis['quality']}</b> ({ema_analysis['score']}/100)\n\n"
            
            caption += (
                f"📊 Pattern: <b>{pattern}</b>\n"
                f"💹 Direzione: <b>{side}</b>\n"
                f"🪙 {symbol} ({timeframe})\n"
                f"🕐 {timestamp_str}\n\n"
                f"💵 Prezzo Entry: ${last_close:.4f}\n"
            )
            
            # Mostra info stop loss
            if USE_EMA_STOP_LOSS:
                caption += f"🛑 Stop Loss: ${sl_price:.4f} (sotto {ema_used})\n"
                if isinstance(ema_value, (int, float)) and ema_value > 0:
                    caption += f"   📍 {ema_used} value: ${ema_value:.4f}\n"
            else:
                caption += f"🛑 Stop Loss: ${sl_price:.4f} (ATR-based)\n"
            
            caption += (
                f"🎯 Take Profit: ${tp_price:.4f}\n"
                f"📦 Qty suggerita: {qty:.4f}\n"
                f"💰 Rischio: ${risk_for_symbol}\n"
                f"📏 R:R = {abs(tp_price-last_close)/abs(sl_price-last_close):.2f}:1"
            )
            
            # Aggiungi dettagli EMA se disponibili
            if ema_analysis and EMA_FILTER_ENABLED:
                caption += f"\n\n📈 <b>EMA Analysis:</b>\n"
                caption += ema_analysis['details']
                
                # Aggiungi reminder sulla strategia EMA SL
                if USE_EMA_STOP_LOSS:
                    caption += f"\n\n💡 <b>Strategia EMA Stop:</b>\n"
                    caption += f"Posizione chiusa se prezzo rompe {ema_used} al ribasso"
            
            if position_exists:
                caption += f"\n\n⚠️ <b>Posizione già aperta per {symbol}</b>"
                caption += f"\nOrdine NON piazzato per evitare duplicati"
            
            # Piazza ordine se autotrade è abilitato E non esiste già posizione
            if job_ctx.get('autotrade') and qty > 0 and not position_exists:
                order_res = await place_bybit_order(symbol, side, qty, sl_price, tp_price)
                
                if 'error' in order_res:
                    if order_res.get('error') == 'position_exists':
                        caption += f"\n\n⚠️ Posizione già aperta, ordine saltato"
                    else:
                        caption += f"\n\n❌ Errore ordine: {order_res['error']}"
                else:
                    caption += f"\n\n✅ Ordine piazzato su Bybit {TRADING_MODE.upper()}"
        else:
            # Nessun pattern trovato (arriviamo qui solo se full_mode è attivo)
            mode_emoji = "🔔" if full_mode else "🔕"
            caption += f"\n{mode_emoji} Nessun pattern rilevato"
            if not math.isnan(last_atr):
                caption += f"\n📏 ATR(14): ${last_atr:.4f}"
            
            # AGGIUNGI ANALISI EMA anche senza pattern (per notifiche complete)
            if full_mode and EMA_FILTER_ENABLED:
                ema_check = analyze_ema_conditions(df, timeframe)
                
                caption += f"\n\n📈 <b>EMA Market Analysis:</b>\n"
                caption += f"Score: {ema_check['score']}/100 ({ema_check['quality']})\n"
                caption += ema_check['details']
                
                # Aggiungi suggerimento basato su quality
                if ema_check['quality'] == 'GOLD':
                    caption += f"\n\n🌟 Setup EMA perfetto! Aspetta pattern qui."
                elif ema_check['quality'] == 'GOOD':
                    caption += f"\n\n✅ Condizioni EMA buone per entry."
                elif ema_check['quality'] == 'OK':
                    caption += f"\n\n⚠️ Condizioni EMA accettabili."
                elif ema_check['quality'] in ['WEAK', 'BAD']:
                    caption += f"\n\n❌ Condizioni EMA sfavorevoli. Evita entry."
            
            caption += f"\n\n💡 Modalità: Notifiche complete attive"
        
        # Genera e invia il grafico
        try:
            chart_buffer = generate_chart(df, symbol, timeframe)
            
            await context.bot.send_photo(
                chat_id=chat_id, 
                photo=chart_buffer, 
                caption=caption,
                parse_mode='HTML'
            )
            
            status = '✅ '+pattern if found else ('🔔 Full mode' if full_mode else '🔕 Default')
            logging.info(f"📸 Grafico inviato per {symbol} {timeframe} - Pattern: {status}")
            
        except Exception as e:
            logging.error(f'Errore generazione/invio grafico: {e}')
            # Se il grafico fallisce, invia almeno il testo
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Errore nel grafico\n\n{caption}",
                parse_mode='HTML'
            )

    except Exception as e:
        logging.exception(f'Errore in analyze_job per {symbol} {timeframe}')
        # Invia errori solo se full mode attivo
        # Ricontrolla perché full_mode potrebbe non essere in scope
        try:
            with FULL_NOTIFICATIONS_LOCK:
                should_send_error = chat_id in FULL_NOTIFICATIONS and key in FULL_NOTIFICATIONS[chat_id]
            
            if should_send_error:
                await context.bot.send_message(
                    chat_id=chat_id, 
                    text=f"❌ Errore nell'analisi di {symbol} {timeframe}: {str(e)}"
                )
        except:
            # Se anche questo fallisce, logga e basta
            logging.error(f'Impossibile inviare messaggio di errore per {symbol} {timeframe}')
    """
    Job che viene eseguito ad ogni chiusura candela
    Se in pausa, invia grafico SOLO quando trova un pattern
    """
    job_ctx = context.job.data
    chat_id = job_ctx['chat_id']
    symbol = job_ctx['symbol']
    timeframe = job_ctx['timeframe']
    key = f'{symbol}-{timeframe}'


    try:
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=200)
        if df.empty:
            logging.warning(f'Nessun dato per {symbol} {timeframe}')
            if not is_paused:  # Invia errore solo se non in pausa
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f'⚠️ Nessun dato disponibile per {symbol} {timeframe}'
                )
            return

        last_close = df['close'].iloc[-1]
        last_time = df.index[-1]
        
        # Controlla pattern
        found, side, pattern = check_patterns(df)
        
        # Se in pausa e NON c'è pattern, non inviare nulla
        if is_paused and not found:
            logging.debug(f'🔇 {symbol} {timeframe} in pausa - nessun pattern, skip notifica')
            return
        
        # Calcola ATR per eventuali SL/TP
        atr_series = atr(df, period=14)
        last_atr = atr_series.iloc[-1] if not atr_series.isna().all() else np.nan
        
        # Prepara messaggio base
        timestamp_str = last_time.strftime('%Y-%m-%d %H:%M UTC')
        caption = (
            f"📊 <b>{symbol}</b> ({timeframe})\n"
            f"🕐 {timestamp_str}\n"
            f"💵 Prezzo: ${last_close:.4f}\n"
        )
        
        # Se c'è volume, mostralo
        if VOLUME_FILTER:
            vol = df['volume']
            if len(vol) >= 21:
                avg_vol = vol.iloc[-21:-1].mean()
                current_vol = vol.iloc[-1]
                vol_ratio = (current_vol / avg_vol) if avg_vol > 0 else 0
                caption += f"📈 Volume: {vol_ratio:.2f}x media\n"
        
        # Se pattern trovato, aggiungi dettagli
        if found:
            logging.info(f'🎯 SEGNALE: {pattern} - {side} su {symbol} {timeframe}')
            
            # Calcola SL e TP
            if not math.isnan(last_atr) and last_atr > 0:
                if side == 'Buy':
                    sl_price = last_close - last_atr * ATR_MULT_SL
                    tp_price = last_close + last_atr * ATR_MULT_TP
                else:
                    sl_price = last_close + last_atr * ATR_MULT_SL
                    tp_price = last_close - last_atr * ATR_MULT_TP
            else:
                # Fallback: usa low/high della candela
                if side == 'Buy':
                    sl_price = df['low'].iloc[-1]
                    tp_price = last_close * 1.02
                else:
                    sl_price = df['high'].iloc[-1]
                    tp_price = last_close * 0.98
            
            # Calcola position size
            qty = calculate_position_size(last_close, sl_price, RISK_USD)
            
            # Usa risk override se disponibile per questo symbol
            risk_for_symbol = SYMBOL_RISK_OVERRIDE.get(symbol, RISK_USD)
            if risk_for_symbol != RISK_USD:
                qty = calculate_position_size(last_close, sl_price, risk_for_symbol)
                logging.info(f'💰 Using risk override for {symbol}: ${risk_for_symbol}')
            
            # Verifica se esiste già una posizione
            position_exists = symbol in ACTIVE_POSITIONS
            
            caption = (
                f"🔥 <b>SEGNALE TROVATO!</b>\n\n"
                f"📊 Pattern: <b>{pattern}</b>\n"
                f"💹 Direzione: <b>{side}</b>\n"
                f"🪙 {symbol} ({timeframe})\n"
                f"🕐 {timestamp_str}\n\n"
                f"💵 Prezzo Entry: ${last_close:.4f}\n"
                f"🛑 Stop Loss: ${sl_price:.4f}\n"
                f"🎯 Take Profit: ${tp_price:.4f}\n"
                f"📦 Qty suggerita: {qty:.4f}\n"
                f"💰 Rischio: ${RISK_USD}\n"
                f"📏 R:R = {abs(tp_price-last_close)/abs(sl_price-last_close):.2f}:1"
            )
            
            if position_exists:
                caption += f"\n\n⚠️ <b>Posizione già aperta per {symbol}</b>"
                caption += f"\nOrdine NON piazzato per evitare duplicati"
            
            # Piazza ordine se autotrade è abilitato E non esiste già posizione
            if job_ctx.get('autotrade') and qty > 0 and not position_exists:
                order_res = await place_bybit_order(symbol, side, qty, sl_price, tp_price)
                
                if 'error' in order_res:
                    if order_res.get('error') == 'position_exists':
                        caption += f"\n\n⚠️ Posizione già aperta, ordine saltato"
                    else:
                        caption += f"\n\n❌ Errore ordine: {order_res['error']}"
                else:
                    caption += f"\n\n✅ Ordine piazzato su Bybit {TRADING_MODE.upper()}"
        else:
            # Nessun pattern trovato
            pause_emoji = "🔇" if is_paused else "⏳"
            caption += f"\n{pause_emoji} Nessun pattern rilevato"
            if not math.isnan(last_atr):
                caption += f"\n📏 ATR(14): ${last_atr:.4f}"
        
        # SEMPRE genera e invia il grafico
        try:
            chart_buffer = generate_chart(df, symbol, timeframe)
            
            await context.bot.send_photo(
                chat_id=chat_id, 
                photo=chart_buffer, 
                caption=caption,
                parse_mode='HTML'
            )
            
            status = '✅ '+pattern if found else ('🔇 Pausa' if is_paused else '❌ Nessuno')
            logging.info(f"📸 Grafico inviato per {symbol} {timeframe} - Pattern: {status}")
            
        except Exception as e:
            logging.error(f'Errore generazione/invio grafico: {e}')
            # Se il grafico fallisce, invia almeno il testo
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Errore nel grafico\n\n{caption}",
                parse_mode='HTML'
            )

    except Exception as e:
        logging.exception(f'Errore in analyze_job per {symbol} {timeframe}')
        if not is_paused:  # Invia errori solo se non in pausa
            await context.bot.send_message(
                chat_id=chat_id, 
                text=f"❌ Errore nell'analisi di {symbol} {timeframe}: {str(e)}"
            )


# ----------------------------- TELEGRAM COMMANDS -----------------------------

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
        "/analizza SYMBOL TF - Inizia analisi\n"
        "/stop SYMBOL - Ferma analisi\n"
        "/list - Analisi attive\n"
        "/abilita SYMBOL TF - Notifiche complete (ogni candela)\n"
        "/pausa SYMBOL TF - Solo pattern (default)\n\n"
        "💼 <b>Comandi Trading:</b>\n"
        "/balance - Mostra saldo\n"
        "/posizioni - Posizioni aperte (sync Bybit)\n"
        "/orders - Ordini chiusi con P&L\n"
        "/sync - Sincronizza tracking con Bybit\n"
        "/chiudi SYMBOL - Rimuovi posizione dal tracking\n\n"
        "🔍 <b>Comandi Debug:</b>\n"
        "/test SYMBOL TF - Test pattern\n\n"
        "🎯 <b>Comandi Pattern:</b>\n"
        "/patterns - Lista pattern e status\n"
        "/pattern_on NOME - Abilita pattern\n"
        "/pattern_off NOME - Disabilita pattern\n"
        "/ema_filter [MODE] - Gestisci filtro EMA\n"
        "/ema_sl [on|off] - EMA Stop Loss\n\n"
        "📝 Esempio: /analizza BTCUSDT 15m\n"
        f"⏱️ Timeframes: {', '.join(ENABLED_TFS)}\n"
        f"💰 Rischio: ${RISK_USD}\n\n"
        "🔕 <b>DEFAULT:</b> Solo notifiche con pattern\n"
        "⚠️ <b>NOTA:</b> Solo segnali BUY attivi"
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
                
                # CORREZIONE: Usa solo HTML sicuro
                msg += f"{side_emoji} <b>{symbol}</b> - {side}\n"
                msg += f"  Qty: {qty:.4f}\n"
                msg += f"  Entry: ${avg_entry:.4f}\n"
                msg += f"  Exit: ${avg_exit:.4f}\n"
                msg += f"  {pnl_emoji} PnL: ${closed_pnl:+.2f} ({pnl_percent:+.2f}%)\n"
                msg += f"  Time: {time_str}\n\n"
            
            # Statistiche finali
            msg += "━━━━━━━━━━━━━━━━━━━\n\n"
            msg += f"💰 <b>PnL Totale: ${total_pnl:+.2f}</b>\n"
            msg += f"✅ Win: {win_count} | ❌ Loss: {loss_count}\n"
            
            if (win_count + loss_count) > 0:
                win_rate = (win_count / (win_count + loss_count)) * 100
                msg += f"📊 Win Rate: {win_rate:.1f}%\n"
            
            msg += f"\n💡 Usa /orders [numero] per vedere più ordini"
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
        found, side, pattern = check_patterns(df)
        
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
            '🆕 Bullish Comeback': is_bullish_comeback(df),
            'Bullish Engulfing': is_bullish_engulfing(prev, last),
            'Bearish Engulfing': is_bearish_engulfing(prev, last),
            'Hammer': is_hammer(last),
            'Shooting Star': is_shooting_star(last),
            'Pin Bar': is_pin_bar(last),
            'Doji': is_doji(last),
            'Morning Star': is_morning_star(prev2, prev, last),
            'Evening Star': is_evening_star(prev2, prev, last),
            'Three White Soldiers': is_three_white_soldiers(prev2, prev, last),
            'Three Black Crows': is_three_black_crows(prev2, prev, last)
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


# ----------------------------- MAIN -----------------------------

def main():
    """Funzione principale"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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
    application.add_handler(CommandHandler('patterns', cmd_patterns))
    application.add_handler(CommandHandler('pattern_on', cmd_pattern_on))
    application.add_handler(CommandHandler('pattern_off', cmd_pattern_off))
    application.add_handler(CommandHandler('pattern_info', cmd_pattern_info))
    application.add_handler(CommandHandler('ema_filter', cmd_ema_filter))
    application.add_handler(CommandHandler('ema_sl', cmd_ema_sl))
    
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

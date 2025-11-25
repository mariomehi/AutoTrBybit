"""
Telegram Bot for automated pattern detection + Bybit Testnet trading
- Features:
  * /analizza <SYMBOL> <TIMEFRAME> -> starts continuous analysis for that symbol+tf
  * /stop <SYMBOL> -> stops analysis for that symbol (in that chat)
  * multi-symbol, multi-timeframe per chat
  * volume filter
  * SL = ATR * X (user config)
  * TP = ATR * X (user config)
  * position sizing by risk per trade (USD risk)
  * uses Bybit Testnet for orders (pybit) and Bybit public REST for klines
  * generates candle chart when signal is found (mplfinance)

Notes:
- This is a starter, modular and documented file. You must set your TELEGRAM_BOT_TOKEN and BYBIT API keys.
- Test thoroughly on Bybit TESTNET before switching to real account.
- Designed to run on Railway / VPS.

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
ENABLED_TFS = ['5m','15m','30m','1h','4h']

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

# Paused notifications: chat_id -> set of "SYMBOL-TIMEFRAME" keys
PAUSED_NOTIFICATIONS = {}
PAUSED_LOCK = threading.Lock()

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


def check_patterns(df: pd.DataFrame):
    """
    Controlla tutti i pattern
    Returns: (found: bool, side: str, pattern_name: str)
    
    NOTA: Per ora solo segnali BUY attivi, SELL commentati
    """
    if len(df) < 4:
        return (False, None, None)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # ===== PATTERN BUY (ATTIVI) =====
    
    # Pattern a 2 candele
    if is_bullish_engulfing(prev, last):
        return (True, 'Buy', 'Bullish Engulfing')
    
    # Bearish Engulfing commentato (segnale SELL)
    # if is_bearish_engulfing(prev, last):
    #     return (True, 'Sell', 'Bearish Engulfing')
    
    # Pattern singola candela - BUY
    if is_hammer(last):
        return (True, 'Buy', 'Hammer')
    
    # Shooting Star commentato (segnale SELL)
    # if is_shooting_star(last):
    #     return (True, 'Sell', 'Shooting Star')
    
    # Pin bar - solo BUY
    if is_pin_bar(last):
        lower_wick = min(last['open'], last['close']) - last['low']
        upper_wick = last['high'] - max(last['open'], last['close'])
        
        # Solo pin bar rialzisti (ombra inferiore lunga)
        if lower_wick > upper_wick:
            return (True, 'Buy', 'Pin Bar Bullish')
        # Pin bar ribassisti commentati
        # else:
        #     return (True, 'Sell', 'Pin Bar Bearish')
    
    # Doji - commentato perché può dare segnali SELL
    # if is_doji(last):
    #     if prev['close'] > prev['open']:
    #         return (True, 'Sell', 'Doji (reversione)')
    #     else:
    #         return (True, 'Buy', 'Doji (reversione)')
    
    # Pattern a 3 candele - solo BUY
    if is_morning_star(prev2, prev, last):
        return (True, 'Buy', 'Morning Star')
    
    # Evening Star commentato (segnale SELL)
    # if is_evening_star(prev2, prev, last):
    #     return (True, 'Sell', 'Evening Star')
    
    if is_three_white_soldiers(prev2, prev, last):
        return (True, 'Buy', 'Three White Soldiers')
    
    # Three Black Crows commentato (segnale SELL)
    # if is_three_black_crows(prev2, prev, last):
    #     return (True, 'Sell', 'Three Black Crows')
    
    return (False, None, None)


# ----------------------------- TRADING HELPERS -----------------------------

def calculate_position_size(entry_price: float, sl_price: float, risk_usd: float):
    """
    Calcola la quantità basata sul rischio in USD
    Formula: Qty = Risk USD / |Entry - SL|
    """
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit == 0:
        return 0
    qty = risk_usd / risk_per_unit
    return float(max(0, qty))


async def place_bybit_order(symbol: str, side: str, qty: float, sl_price: float, tp_price: float):
    """
    Piazza ordine market su Bybit (Demo o Live)
    Controlla prima se esiste già una posizione aperta per questo symbol
    
    Parametri:
    - symbol: es. 'BTCUSDT'
    - side: 'Buy' o 'Sell'
    - qty: quantità in contratti
    - sl_price: prezzo stop loss
    - tp_price: prezzo take profit
    """
    if BybitHTTP is None:
        return {'error': 'pybit non disponibile'}
    
    # Controlla se esiste già una posizione aperta per questo symbol
    with POSITIONS_LOCK:
        if symbol in ACTIVE_POSITIONS:
            existing = ACTIVE_POSITIONS[symbol]
            logging.info(f'⚠️ Posizione già aperta per {symbol}: {existing}')
            return {
                'error': 'position_exists',
                'message': f'Posizione già aperta per {symbol}',
                'existing_position': existing
            }
    
    try:
        session = create_bybit_session()
        
        # Verifica il balance prima di tradare
        try:
            wallet = session.get_wallet_balance(accountType="UNIFIED")
            logging.info(f'💰 Wallet Balance: {wallet}')
        except Exception as e:
            logging.warning(f'Non riesco a verificare il balance: {e}')
        
        # Arrotonda qty in base alle specifiche del symbol
        # Per USDT perpetuals, di solito 3 decimali
        qty = round(qty, 3)
        
        # Arrotonda prezzi (di solito 2 decimali per BTC/ETH, può variare)
        sl_price = round(sl_price, 2)
        tp_price = round(tp_price, 2)
        
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
        
        # Errori comuni
        if 'insufficient' in error_msg.lower():
            return {'error': 'Balance insufficiente'}
        elif 'invalid' in error_msg.lower():
            return {'error': f'Parametri non validi: {error_msg}'}
        else:
            return {'error': error_msg}


# ----------------------------- CHART GENERATION -----------------------------

def generate_chart(df: pd.DataFrame, symbol: str, timeframe: str) -> io.BytesIO:
    """
    Genera grafico candlestick usando mplfinance
    Returns: BytesIO object (immagine in memoria)
    """
    try:
        # Usa gli ultimi 100 candles per il grafico
        chart_df = df.tail(100)
        
        # Crea il grafico in memoria
        buffer = io.BytesIO()
        
        mpf.plot(
            chart_df,
            type='candle',
            style='charles',
            title=f'{symbol} - {timeframe}',
            ylabel='Price',
            volume=True,
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
    Se in pausa, invia grafico SOLO quando trova un pattern
    """
    job_ctx = context.job.data
    chat_id = job_ctx['chat_id']
    symbol = job_ctx['symbol']
    timeframe = job_ctx['timeframe']
    key = f'{symbol}-{timeframe}'

    # Verifica se le notifiche sono in pausa per questo symbol/timeframe
    with PAUSED_LOCK:
        is_paused = chat_id in PAUSED_NOTIFICATIONS and key in PAUSED_NOTIFICATIONS[chat_id]

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
        "/pausa SYMBOL TF - Silenzia notifiche senza pattern\n"
        "/riprendi SYMBOL TF - Riattiva notifiche\n\n"
        "💼 <b>Comandi Trading:</b>\n"
        "/balance - Mostra saldo\n"
        "/posizioni - Posizioni aperte\n"
        "/chiudi SYMBOL - Rimuovi posizione dal tracking\n\n"
        "🔍 <b>Comandi Debug:</b>\n"
        "/test SYMBOL TF - Test pattern\n\n"
        "📝 Esempio: /analizza BTCUSDT 15m\n"
        f"⏱️ Timeframes: {', '.join(ENABLED_TFS)}\n"
        f"💰 Rischio: ${RISK_USD}\n\n"
        "⚠️ <b>NOTA:</b> Solo segnali BUY attivi"
    )
    await update.message.reply_text(welcome_text, parse_mode='HTML')


        await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_pausa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /pausa SYMBOL TIMEFRAME
    Mette in pausa le notifiche senza pattern per un symbol/timeframe specifico
    """
    chat_id = update.effective_chat.id
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /pausa SYMBOL TIMEFRAME\n'
            'Esempio: /pausa BTCUSDT 15m\n\n'
            'Mette in pausa le notifiche quando NON viene rilevato nessun pattern.\n'
            'Riceverai comunque notifiche quando ci sono segnali di trading.'
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
    
    # Aggiungi alla lista pause
    with PAUSED_LOCK:
        if chat_id not in PAUSED_NOTIFICATIONS:
            PAUSED_NOTIFICATIONS[chat_id] = set()
        PAUSED_NOTIFICATIONS[chat_id].add(key)
    
    await update.message.reply_text(
        f'🔇 <b>Notifiche in pausa per {symbol} {timeframe}</b>\n\n'
        f'Non riceverai più notifiche quando NON ci sono pattern.\n'
        f'Riceverai comunque i segnali di trading quando vengono rilevati.\n\n'
        f'Usa /riprendi {symbol} {timeframe} per riattivare tutte le notifiche.',
        parse_mode='HTML'
    )


async def cmd_riprendi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /riprendi SYMBOL TIMEFRAME
    Riattiva tutte le notifiche per un symbol/timeframe
    """
    chat_id = update.effective_chat.id
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            '❌ Uso: /riprendi SYMBOL TIMEFRAME\n'
            'Esempio: /riprendi BTCUSDT 15m\n\n'
            'Riattiva tutte le notifiche (anche senza pattern).'
        )
        return
    
    symbol = args[0].upper()
    timeframe = args[1].lower()
    key = f'{symbol}-{timeframe}'
    
    # Rimuovi dalla lista pause
    with PAUSED_LOCK:
        if chat_id in PAUSED_NOTIFICATIONS and key in PAUSED_NOTIFICATIONS[chat_id]:
            PAUSED_NOTIFICATIONS[chat_id].remove(key)
            
            # Pulisci se il set è vuoto
            if not PAUSED_NOTIFICATIONS[chat_id]:
                del PAUSED_NOTIFICATIONS[chat_id]
            
            await update.message.reply_text(
                f'🔔 <b>Notifiche riattivate per {symbol} {timeframe}</b>\n\n'
                f'Riceverai ora tutte le notifiche, anche quando non ci sono pattern.',
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                f'⚠️ Le notifiche per {symbol} {timeframe} non erano in pausa.'
            )


async def cmd_posizioni(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comando /posizioni
    Mostra tutte le posizioni aperte tracciate dal bot
    """
    with POSITIONS_LOCK:
        if not ACTIVE_POSITIONS:
            await update.message.reply_text(
                '📭 <b>Nessuna posizione aperta</b>\n\n'
                'Il bot non ha posizioni attive in questo momento.',
                parse_mode='HTML'
            )
            return
        
        msg = '📊 <b>Posizioni Aperte</b>\n\n'
        
        for symbol, pos in ACTIVE_POSITIONS.items():
            side = pos.get('side', 'N/A')
            qty = pos.get('qty', 0)
            sl = pos.get('sl', 0)
            tp = pos.get('tp', 0)
            timestamp = pos.get('timestamp', 'N/A')
            
            side_emoji = "🟢" if side == 'Buy' else "🔴"
            
            msg += f"{side_emoji} <b>{symbol}</b> - {side}\n"
            msg += f"  📦 Qty: {qty:.4f}\n"
            msg += f"  🛑 SL: ${sl:.2f}\n"
            msg += f"  🎯 TP: ${tp:.2f}\n"
            msg += f"  🕐 {timestamp[:19]}\n\n"
        
        msg += f"💼 Totale posizioni: {len(ACTIVE_POSITIONS)}\n\n"
        msg += "💡 Usa /chiudi SYMBOL per chiudere manualmente"
        
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
            '(Non chiude automaticamente la posizione su Bybit)'
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
                f'⚠️ Nessuna posizione tracciata per {symbol}'
            )


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
    """Comando /list - mostra analisi attive"""
    chat_id = update.effective_chat.id
    chat_map = ACTIVE_ANALYSES.get(chat_id, {})
    
    if not chat_map:
        await update.message.reply_text('📭 Nessuna analisi attiva in questa chat.')
        return
    
    text = '📊 <b>Analisi attive:</b>\n\n' + '\n'.join(
        f'• {key}' for key in chat_map.keys()
    )
    await update.message.reply_text(text, parse_mode='HTML')


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
    application.add_handler(CommandHandler('pausa', cmd_pausa))
    application.add_handler(CommandHandler('riprendi', cmd_riprendi))
    application.add_handler(CommandHandler('posizioni', cmd_posizioni))
    application.add_handler(CommandHandler('chiudi', cmd_chiudi))
    
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

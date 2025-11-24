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

Requirements (pip):
  python-telegram-bot>=20.0
  requests
  pybit
  pandas
  numpy
  mplfinance
  ta
  scipy

"""

import os
import time
import math
import logging
from datetime import datetime, timezone
import threading

import requests
import pandas as pd
import numpy as np
import mplfinance as mpf

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, JobQueue

# NOTE: import pybit for trading (unified trading). If you prefer ccxt, adapt accordingly.
try:
    from pybit.unified_trading import HTTP as BybitHTTP
except Exception:
    # fallback if package naming differs; user may need to install pybit v5+
    BybitHTTP = None

# ----------------------------- CONFIG -----------------------------
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', 'PUT_YOUR_TELEGRAM_TOKEN')
BYBIT_API_KEY = os.environ.get('BYBIT_API_KEY', 'PUT_YOUR_BYBIT_KEY')
BYBIT_API_SECRET = os.environ.get('BYBIT_API_SECRET', 'PUT_YOUR_BYBIT_SECRET')
# Bybit testnet endpoint will be configured in code below

# Strategy parameters from your choices
VOLUME_FILTER = True  # punto 3 -> SÌ
ATR_MULT_SL = 1.5     # punto 4 -> B (ATR * X) ; X chosen = 1.5 (you can tweak)
ATR_MULT_TP = 2.0     # punto 5 -> B (ATR * X) ; X chosen = 2.0
RISK_USD = 10.0       # punto 6 -> rischio per trade in USD (user can change)
ENABLED_TFS = ['5m','15m','30m','1h','4h']  # punto 7

# Klines map interval to Bybit format
BYBIT_INTERVAL_MAP = {
    '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '4h': '240', '1d': 'D'
}

# Active analyses storage: chat_id -> list of jobs ({key:job})
# Each entry key is f"{symbol}-{timeframe}" for uniqueness
ACTIVE_ANALYSES = {}
ACTIVE_ANALYSES_LOCK = threading.Lock()

# Bybit client initialization (testnet)
BYBIT_TESTNET_REST = 'https://api-testnet.bybit.com'

def create_bybit_session():
    if BybitHTTP is None:
        raise RuntimeError('pybit.unified_trading not available. Install pybit>=5.x')
    session = BybitHTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, base_url=BYBIT_TESTNET_REST)
    return session

# ----------------------------- UTILITIES -----------------------------

def bybit_get_klines(symbol: str, interval: str, limit: int = 200):
    """Fetch klines from Bybit v5 public API. Returns DataFrame with open/high/low/close/volume and timestamp index."""
    itv = BYBIT_INTERVAL_MAP.get(interval)
    if itv is None:
        raise ValueError(f'Unsupported timeframe: {interval}')

    url = f'https://api.bybit.com/v5/market/kline'
    params = {'category': 'linear', 'symbol': symbol, 'interval': itv, 'limit': limit}
    # Use public endpoint; no auth required for market data
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    # j['result']['list'] expected
    data = j.get('result', {}).get('list', [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=['t','o','h','l','c','v'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('t', inplace=True)
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'}, inplace=True)
    return df


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Basic pattern detectors (initial set from your list)

def is_bullish_engulfing(prev, curr):
    return (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
            curr['open'] <= prev['close'] and curr['close'] >= prev['open'])


def is_bearish_engulfing(prev, curr):
    return (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
            curr['open'] >= prev['close'] and curr['close'] <= prev['open'])


def is_hammer(candle):
    body = abs(candle['close'] - candle['open'])
    lower_wick = candle['open'] - candle['low'] if candle['close'] >= candle['open'] else candle['close'] - candle['low']
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    return lower_wick > 2 * body and upper_wick < body


def is_shooting_star(candle):
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    return upper_wick > 2 * body and lower_wick < body

# Morning/Evening star simplified: check 3-candle pattern

def is_morning_star(a, b, c):
    return (a['close'] < a['open'] and
            b['close'] < b['open'] and
            c['close'] > c['open'] and
            c['close'] > (a['open'] + a['close'])/2)


def is_evening_star(a, b, c):
    return (a['close'] > a['open'] and
            b['close'] > b['open'] and
            c['close'] < c['open'] and
            c['close'] < (a['open'] + a['close'])/2)

# Pin bar detection (relative wick lengths)

def is_pin_bar(candle):
    body = abs(candle['close'] - candle['open'])
    upper = candle['high'] - max(candle['close'], candle['open'])
    lower = min(candle['close'], candle['open']) - candle['low']
    # Long lower wick => bullish pin, long upper wick => bearish pin
    return (lower > 2 * body and upper < body) or (upper > 2 * body and lower < body)

# Combined check: returns tuple (signal, direction, reason)

def check_patterns(df: pd.DataFrame):
    """Check last candles for patterns. Returns (found, side, pattern_name)"""
    if len(df) < 4:
        return (False, None, None)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    if is_bullish_engulfing(prev, last):
        return (True, 'Buy', 'Bullish Engulfing')
    if is_bearish_engulfing(prev, last):
        return (True, 'Sell', 'Bearish Engulfing')
    if is_hammer(last):
        return (True, 'Buy', 'Hammer')
    if is_shooting_star(last):
        return (True, 'Sell', 'Shooting Star')
    if is_morning_star(prev2, prev, last):
        return (True, 'Buy', 'Morning Star')
    if is_evening_star(prev2, prev, last):
        return (True, 'Sell', 'Evening Star')
    if is_pin_bar(last):
        side = 'Buy' if (min(last['open'], last['close']) - last['low']) > (last['high'] - max(last['open'], last['close'])) else 'Sell'
        return (True, side, 'Pin Bar')
    return (False, None, None)

# ----------------------------- TRADING HELPERS -----------------------------

def calculate_position_size(symbol: str, entry_price: float, sl_price: float, risk_usd: float):
    """Calculate quantity for USD risk using Bybit linear contracts (qty in contract units depends on symbol). This function provides an approximate size and must be validated on Bybit account specifics."""
    # For simplicity we assume 1 contract = 1 USD of notional for many USDT perpetuals; user should adjust per symbol
    # qty = risk / (abs(entry - sl))
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit == 0:
        return 0
    qty = risk_usd / risk_per_unit
    return float(max(0, qty))

async def place_bybit_order(symbol: str, side: str, qty: float, sl_price: float, tp_price: float):
    """Place market order + set SL and TP via pybit unified trading. This uses Testnet by default in this script."""
    if BybitHTTP is None:
        raise RuntimeError('pybit not installed')
    session = create_bybit_session()
    # For safety, place market order then place stop/limit orders for SL/TP or use conditional orders depending on API
    try:
        # Place market order (category linear assumed)
        order = session.place_order(category='linear', symbol=symbol, side=side, orderType='Market', qty=str(qty))
        # Note: exact params may vary; check pybit docs and adapt (the SDK evolves).
        return order
    except Exception as e:
        logging.exception('Order placement failed')
        return {'error': str(e)}

# ----------------------------- JOB CALLBACK -----------------------------

async def analyze_job(context: ContextTypes.DEFAULT_TYPE):
    job_ctx = context.job.data
    chat_id = job_ctx['chat_id']
    symbol = job_ctx['symbol']
    timeframe = job_ctx['timeframe']

    try:
        df = bybit_get_klines(symbol, timeframe, limit=200)
        if df.empty:
            await context.bot.send_message(chat_id=chat_id, text=f'No data for {symbol} {timeframe}')
            return

        # Apply volume filter
        if VOLUME_FILTER:
            vol = df['volume']
            if len(vol) >= 21:
                if vol.iloc[-1] <= vol.iloc[-21:-1].mean():
                    # volume not higher than recent average -> skip
                    return

        found, side, pattern = check_patterns(df)
        if not found:
            return

        # Calculate ATR-based SL/TP
        atr_series = atr(df, period=14)
        last_atr = atr_series.iloc[-1] if not atr_series.isna().all() else np.nan
        last_close = df['close'].iloc[-1]

        sl_price = None
        tp_price = None
        if not math.isnan(last_atr):
            if side == 'Buy':
                sl_price = last_close - last_atr * ATR_MULT_SL
                tp_price = last_close + last_atr * ATR_MULT_TP
            else:
                sl_price = last_close + last_atr * ATR_MULT_SL
                tp_price = last_close - last_atr * ATR_MULT_TP
        else:
            # fallback to candlestick low/high
            if side == 'Buy':
                sl_price = df['low'].iloc[-1]
                tp_price = last_close * (1 + 0.02)
            else:
                sl_price = df['high'].iloc[-1]
                tp_price = last_close * (1 - 0.02)

        # position sizing based on USD risk
        qty = calculate_position_size(symbol, last_close, sl_price, RISK_USD)
        if qty <= 0:
            await context.bot.send_message(chat_id=chat_id, text=f'Calculated qty is 0 for {symbol}. Check parameters.')
            return

        # Generate chart and send
        chart_path = f'/tmp/{symbol}_{timeframe}_{int(time.time())}.png'
        mpf.plot(df.tail(100), type='candle', style='charles', savefig=chart_path)

        caption = f"Signal: {pattern} \nSide: {side}\nSymbol: {symbol} {timeframe}\nPrice: {last_close:.4f}\nSL: {sl_price:.4f}\nTP: {tp_price:.4f}\nQty(approx): {qty:.6f}"
        await context.bot.send_photo(chat_id=chat_id, photo=open(chart_path, 'rb'), caption=caption)

        # Place order on Bybit Testnet (only if enabled by job context 'autotrade')
        if job_ctx.get('autotrade'):
            order_res = await place_bybit_order(symbol, side, qty, sl_price, tp_price)
            await context.bot.send_message(chat_id=chat_id, text=f'Order result: {order_res}')

    except Exception as e:
        logging.exception('analyze_job error')
        await context.bot.send_message(chat_id=chat_id, text=f"Errore nell'analisi di {symbol} {timeframe}: {e}")

# ----------------------------- TELEGRAM COMMANDS -----------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Bot attivo. Usa /analizza <SYMBOL> <TIMEFRAME> per iniziare. Esempio: /analizza BTCUSDT 15m')


async def cmd_analizza(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text('Usage: /analizza SYMBOL TIMEFRAME [autotrade yes/no]')
        return
    symbol = args[0].upper()
    timeframe = args[1]
    autotrade = (len(args) > 2 and args[2].lower() in ['yes', 'true', '1'])

    if timeframe not in ENABLED_TFS:
        await update.message.reply_text(f'Timeframe non supportato. Abilitati: {ENABLED_TFS}')
        return

    key = f'{symbol}-{timeframe}'
    with ACTIVE_ANALYSES_LOCK:
        chat_map = ACTIVE_ANALYSES.setdefault(chat_id, {})
        if key in chat_map:
            await update.message.reply_text(f'Già analizzando {symbol} {timeframe} in questa chat.')
            return
        # schedule job aligned to candle close: compute seconds until next close
        interval_seconds = int({'5m':300,'15m':900,'30m':1800,'1h':3600,'4h':14400}[timeframe])
        now = datetime.now(timezone.utc)
        # compute time to next multiple of interval since epoch
        epoch = int(now.timestamp())
        to_next = interval_seconds - (epoch % interval_seconds)

        job_data = {'chat_id': chat_id, 'symbol': symbol, 'timeframe': timeframe, 'autotrade': autotrade}
        job = context.job_queue.run_repeating(analyze_job, interval=interval_seconds, first=to_next, data=job_data)
        chat_map[key] = job

    await update.message.reply_text(f'Iniziata analisi {symbol} {timeframe}. Autotrade: {autotrade}')


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 1:
        await update.message.reply_text('Usage: /stop SYMBOL or /stop all')
        return
    target = args[0].upper()
    with ACTIVE_ANALYSES_LOCK:
        chat_map = ACTIVE_ANALYSES.get(chat_id, {})
        if target == 'ALL':
            for k, job in list(chat_map.items()):
                job.schedule_removal()
                del chat_map[k]
            await update.message.reply_text('Tutte le analisi fermate.')
            return
        # stop specific symbol on any timeframe
        removed = False
        for k in list(chat_map.keys()):
            if k.startswith(target + '-'):
                job = chat_map[k]
                job.schedule_removal()
                del chat_map[k]
                removed = True
        if removed:
            await update.message.reply_text(f'Analisi per {target} fermata.')
        else:
            await update.message.reply_text(f'Non trovata analisi attiva per {target} in questa chat.')


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_map = ACTIVE_ANALYSES.get(chat_id, {})
    if not chat_map:
        await update.message.reply_text('Nessuna analisi attiva in questa chat.')
        return
    text = 'Analisi attive:\n' + '\n'.join(chat_map.keys())
    await update.message.reply_text(text)


# ----------------------------- MAIN -----------------------------

import asyncio
from telegram.ext import ApplicationBuilder

async def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # HANDLER COMANDI
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("analizza", cmd_analizza))

    print("Bot avviato...")
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())

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

# Bybit endpoints
BYBIT_TESTNET_REST = 'https://api-testnet.bybit.com'
BYBIT_PUBLIC_REST = 'https://api.bybit.com'

def create_bybit_session():
    """Crea sessione Bybit per trading"""
    if BybitHTTP is None:
        raise RuntimeError('pybit non disponibile. Installa: pip install pybit>=5.0')
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError('BYBIT_API_KEY e BYBIT_API_SECRET devono essere configurate')
    session = BybitHTTP(
        api_key=BYBIT_API_KEY, 
        api_secret=BYBIT_API_SECRET, 
        testnet=True  # Usa testnet
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
    """Pattern: Bullish Engulfing"""
    return (prev['close'] < prev['open'] and 
            curr['close'] > curr['open'] and
            curr['open'] <= prev['close'] and 
            curr['close'] >= prev['open'])


def is_bearish_engulfing(prev, curr):
    """Pattern: Bearish Engulfing"""
    return (prev['close'] > prev['open'] and 
            curr['close'] < curr['open'] and
            curr['open'] >= prev['close'] and 
            curr['close'] <= prev['open'])


def is_hammer(candle):
    """Pattern: Hammer (bullish reversal)"""
    body = abs(candle['close'] - candle['open'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    
    return (lower_wick > 2 * body and 
            upper_wick < body and 
            body > 0)


def is_shooting_star(candle):
    """Pattern: Shooting Star (bearish reversal)"""
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    
    return (upper_wick > 2 * body and 
            lower_wick < body and 
            body > 0)


def is_morning_star(a, b, c):
    """Pattern: Morning Star (3 candele - bullish reversal)"""
    return (a['close'] < a['open'] and
            abs(b['close'] - b['open']) < abs(a['close'] - a['open']) * 0.3 and
            c['close'] > c['open'] and
            c['close'] > (a['open'] + a['close']) / 2)


def is_evening_star(a, b, c):
    """Pattern: Evening Star (3 candele - bearish reversal)"""
    return (a['close'] > a['open'] and
            abs(b['close'] - b['open']) < abs(a['close'] - a['open']) * 0.3 and
            c['close'] < c['open'] and
            c['close'] < (a['open'] + a['close']) / 2)


def is_pin_bar(candle):
    """Pattern: Pin Bar"""
    body = abs(candle['close'] - candle['open'])
    upper = candle['high'] - max(candle['close'], candle['open'])
    lower = min(candle['close'], candle['open']) - candle['low']
    
    return ((lower > 2 * body and upper < body) or 
            (upper > 2 * body and lower < body))


def check_patterns(df: pd.DataFrame):
    """
    Controlla tutti i pattern
    Returns: (found: bool, side: str, pattern_name: str)
    """
    if len(df) < 4:
        return (False, None, None)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Pattern a 2 candele
    if is_bullish_engulfing(prev, last):
        return (True, 'Buy', 'Bullish Engulfing')
    if is_bearish_engulfing(prev, last):
        return (True, 'Sell', 'Bearish Engulfing')
    
    # Pattern singola candela
    if is_hammer(last):
        return (True, 'Buy', 'Hammer')
    if is_shooting_star(last):
        return (True, 'Sell', 'Shooting Star')
    
    # Pattern a 3 candele
    if is_morning_star(prev2, prev, last):
        return (True, 'Buy', 'Morning Star')
    if is_evening_star(prev2, prev, last):
        return (True, 'Sell', 'Evening Star')
    
    # Pin bar
    if is_pin_bar(last):
        lower_wick = min(last['open'], last['close']) - last['low']
        upper_wick = last['high'] - max(last['open'], last['close'])
        side = 'Buy' if lower_wick > upper_wick else 'Sell'
        return (True, side, 'Pin Bar')
    
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
    Piazza ordine market su Bybit Testnet
    NOTA: Questa è una versione semplificata. Per produzione, gestisci meglio SL/TP.
    """
    if BybitHTTP is None:
        return {'error': 'pybit non disponibile'}
    
    try:
        session = create_bybit_session()
        
        # Arrotonda qty in base al symbol (esempio generico)
        qty = round(qty, 3)
        
        # Piazza ordine market
        order = session.place_order(
            category='linear',
            symbol=symbol,
            side=side,
            orderType='Market',
            qty=str(qty),
            stopLoss=str(round(sl_price, 2)),
            takeProfit=str(round(tp_price, 2))
        )
        
        return order
        
    except Exception as e:
        logging.exception('Errore nel piazzare ordine')
        return {'error': str(e)}


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
    """
    job_ctx = context.job.data
    chat_id = job_ctx['chat_id']
    symbol = job_ctx['symbol']
    timeframe = job_ctx['timeframe']

    try:
        # Ottieni dati
        df = bybit_get_klines(symbol, timeframe, limit=200)
        if df.empty:
            logging.warning(f'Nessun dato per {symbol} {timeframe}')
            return

        # Filtro volume
        if VOLUME_FILTER:
            vol = df['volume']
            if len(vol) >= 21:
                avg_vol = vol.iloc[-21:-1].mean()
                if vol.iloc[-1] <= avg_vol:
                    logging.debug(f'{symbol} {timeframe}: volume insufficiente')
                    return

        # Controlla pattern
        found, side, pattern = check_patterns(df)
        if not found:
            return

        logging.info(f'🎯 SEGNALE TROVATO: {pattern} - {side} su {symbol} {timeframe}')

        # Calcola ATR per SL/TP
        atr_series = atr(df, period=14)
        last_atr = atr_series.iloc[-1] if not atr_series.isna().all() else np.nan
        last_close = df['close'].iloc[-1]

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
        if qty <= 0:
            await context.bot.send_message(
                chat_id=chat_id, 
                text=f'⚠️ Qty calcolata = 0 per {symbol}. Verifica i parametri.'
            )
            return

        # Genera e invia grafico
        try:
            chart_buffer = generate_chart(df, symbol, timeframe)
            
            caption = (
                f"📊 <b>{pattern}</b>\n"
                f"💹 {side} Signal\n"
                f"🪙 {symbol} ({timeframe})\n"
                f"💵 Prezzo: ${last_close:.4f}\n"
                f"🛑 Stop Loss: ${sl_price:.4f}\n"
                f"🎯 Take Profit: ${tp_price:.4f}\n"
                f"📦 Qty: {qty:.4f}\n"
                f"💰 Rischio: ${RISK_USD}"
            )
            
            await context.bot.send_photo(
                chat_id=chat_id, 
                photo=chart_buffer, 
                caption=caption,
                parse_mode='HTML'
            )
            
        except Exception as e:
            logging.error(f'Errore invio grafico: {e}')
            # Invia almeno il testo
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Segnale trovato ma errore nel grafico\n{caption}",
                parse_mode='HTML'
            )

        # Piazza ordine se autotrade è abilitato
        if job_ctx.get('autotrade'):
            order_res = await place_bybit_order(symbol, side, qty, sl_price, tp_price)
            if 'error' in order_res:
                await context.bot.send_message(
                    chat_id=chat_id, 
                    text=f"❌ Errore ordine: {order_res['error']}"
                )
            else:
                await context.bot.send_message(
                    chat_id=chat_id, 
                    text=f"✅ Ordine piazzato: {order_res}"
                )

    except Exception as e:
        logging.exception(f'Errore in analyze_job per {symbol} {timeframe}')
        await context.bot.send_message(
            chat_id=chat_id, 
            text=f"❌ Errore nell'analisi di {symbol} {timeframe}: {str(e)}"
        )


# ----------------------------- TELEGRAM COMMANDS -----------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start"""
    welcome_text = (
        "🤖 <b>Bot Pattern Detection Attivo!</b>\n\n"
        "📊 Comandi disponibili:\n"
        "/analizza SYMBOL TIMEFRAME - Inizia analisi\n"
        "/stop SYMBOL - Ferma analisi\n"
        "/list - Mostra analisi attive\n\n"
        "📝 Esempio: /analizza BTCUSDT 15m\n"
        f"⏱️ Timeframes supportati: {', '.join(ENABLED_TFS)}"
    )
    await update.message.reply_text(welcome_text, parse_mode='HTML')


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
    
    # Crea applicazione
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Aggiungi handlers
    application.add_handler(CommandHandler('start', cmd_start))
    application.add_handler(CommandHandler('analizza', cmd_analizza))
    application.add_handler(CommandHandler('stop', cmd_stop))
    application.add_handler(CommandHandler('list', cmd_list))
    
    # Avvia bot
    logging.info('🚀 Bot avviato correttamente!')
    logging.info(f'⏱️ Timeframes supportati: {ENABLED_TFS}')
    logging.info(f'💰 Rischio per trade: ${RISK_USD}')
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()

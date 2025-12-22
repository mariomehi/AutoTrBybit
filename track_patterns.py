# ----------------------------- PATTERN STATISTICS SYSTEM -----------------------------

"""
Pattern Statistics Tracking System
Traccia performance di ogni pattern: win rate, PnL, best/worst trade, etc.
"""

import json
import logging
import threading
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

# ← AGGIUNGI QUESTI IMPORT (servono per le integrazioni):
from typing import Optional, Dict, Any

# ← AGGIUNGI QUESTI IMPORT:
try:
    from telegram import Update
    from telegram.ext import ContextTypes
except ImportError:
    logging.warning("telegram imports not available - commands will not work")
    Update = None
    ContextTypes = None

# Storage per statistiche pattern (in memoria + file)
PATTERN_STATS = defaultdict(lambda: {
    'total_signals': 0,
    'total_trades': 0,
    'wins': 0,
    'losses': 0,
    'total_pnl': 0.0,
    'avg_pnl': 0.0,
    'win_rate': 0.0,
    'best_trade': 0.0,
    'worst_trade': 0.0,
    'total_volume': 0.0,
    'avg_duration_minutes': 0.0,
    'by_timeframe': {},
    'by_symbol': {},
    'last_updated': None
})

PATTERN_STATS_LOCK = threading.Lock()
STATS_FILE_PATH = Path('/tmp/pattern_stats.json')  # Railway-compatible3

def load_pattern_stats():
    """Carica statistiche pattern da file"""
    try:
        if STATS_FILE_PATH.exists():
            with open(STATS_FILE_PATH, 'r') as f:
                data = json.load(f)
                
                with PATTERN_STATS_LOCK:
                    PATTERN_STATS.clear()
                    for pattern, stats in data.items():
                        PATTERN_STATS[pattern] = stats
                
                logging.info(f'✅ Statistiche pattern caricate: {len(data)} pattern')
                return True
    except Exception as e:
        logging.error(f'Errore caricamento statistiche: {e}')
    
    return False


def save_pattern_stats():
    """Salva statistiche pattern su file"""
    try:
        with PATTERN_STATS_LOCK:
            data = dict(PATTERN_STATS)
        
        # Crea directory se non esiste
        STATS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(STATS_FILE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.debug(f'💾 Statistiche salvate: {len(data)} pattern')
        return True
    except Exception as e:
        logging.error(f'Errore salvataggio statistiche: {e}')
        return False


def record_pattern_signal(
    pattern_name: str,
    symbol: str,
    timeframe: str,
    side: str,
    traded: bool = False
):
    """
    Registra un segnale pattern
    
    Args:
        pattern_name: Nome pattern (es. "Bullish Engulfing")
        symbol: Symbol (es. "BTCUSDT")
        timeframe: Timeframe (es. "5m")
        side: 'Buy' o 'Sell'
        traded: True se ordine piazzato, False se solo segnale
    """
    try:
        with PATTERN_STATS_LOCK:
            stats = PATTERN_STATS[pattern_name]
            
            # Incrementa contatori globali
            stats['total_signals'] += 1
            if traded:
                stats['total_trades'] += 1
            
            # Stats per timeframe
            if timeframe not in stats['by_timeframe']:
                stats['by_timeframe'][timeframe] = {
                    'signals': 0,
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'pnl': 0.0
                }
            
            stats['by_timeframe'][timeframe]['signals'] += 1
            if traded:
                stats['by_timeframe'][timeframe]['trades'] += 1
            
            # Stats per symbol
            if symbol not in stats['by_symbol']:
                stats['by_symbol'][symbol] = {
                    'signals': 0,
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'pnl': 0.0
                }
            
            stats['by_symbol'][symbol]['signals'] += 1
            if traded:
                stats['by_symbol'][symbol]['trades'] += 1
            
            # Timestamp
            stats['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Salva su file (asincrono per non bloccare)
        save_pattern_stats()
        
        logging.debug(
            f"📊 Pattern signal recorded: {pattern_name} - {symbol} {timeframe} "
            f"({side}) - Traded: {traded}"
        )
        
    except Exception as e:
        logging.error(f'Errore record_pattern_signal: {e}')


def record_pattern_trade_result(
    pattern_name: str,
    symbol: str,
    timeframe: str,
    side: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    duration_minutes: int
):
    """
    Registra il risultato di un trade basato su pattern
    
    Args:
        pattern_name: Nome pattern
        symbol: Symbol
        timeframe: Timeframe
        side: 'Buy' o 'Sell'
        entry_price: Prezzo entrata
        exit_price: Prezzo uscita
        pnl: Profit/Loss in USD
        duration_minutes: Durata trade in minuti
    """
    try:
        with PATTERN_STATS_LOCK:
            stats = PATTERN_STATS[pattern_name]
            
            # Determina win/loss
            is_win = pnl > 0
            
            # Update global stats
            if is_win:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            
            stats['total_pnl'] += pnl
            
            # Win rate
            total_completed = stats['wins'] + stats['losses']
            if total_completed > 0:
                stats['win_rate'] = (stats['wins'] / total_completed) * 100
                stats['avg_pnl'] = stats['total_pnl'] / total_completed
            
            # Best/Worst trade
            if pnl > stats['best_trade']:
                stats['best_trade'] = pnl
            if pnl < stats['worst_trade']:
                stats['worst_trade'] = pnl
            
            # Volume traded
            volume = abs(exit_price - entry_price)
            stats['total_volume'] += volume
            
            # Avg duration
            if stats['avg_duration_minutes'] == 0:
                stats['avg_duration_minutes'] = duration_minutes
            else:
                # Moving average
                stats['avg_duration_minutes'] = (
                    stats['avg_duration_minutes'] * 0.9 + duration_minutes * 0.1
                )
            
            # Update per timeframe
            tf_stats = stats['by_timeframe'].get(timeframe)
            if tf_stats:
                if is_win:
                    tf_stats['wins'] += 1
                else:
                    tf_stats['losses'] += 1
                tf_stats['pnl'] += pnl
            
            # Update per symbol
            sym_stats = stats['by_symbol'].get(symbol)
            if sym_stats:
                if is_win:
                    sym_stats['wins'] += 1
                else:
                    sym_stats['losses'] += 1
                sym_stats['pnl'] += pnl
            
            stats['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Salva
        save_pattern_stats()
        
        result_emoji = "✅" if is_win else "❌"
        logging.info(
            f"{result_emoji} Pattern trade result: {pattern_name} - "
            f"{symbol} {timeframe} - PnL: ${pnl:+.2f}"
        )
        
    except Exception as e:
        logging.error(f'Errore record_pattern_trade_result: {e}')


def get_pattern_stats_summary() -> str:
    """
    Genera report testuale con statistiche pattern
    
    Returns:
        str: Report formattato per Telegram
    """
    try:
        with PATTERN_STATS_LOCK:
            data = dict(PATTERN_STATS)
        
        if not data:
            return "📊 <b>Nessuna statistica disponibile</b>\n\nI pattern non hanno ancora generato trade."
        
        msg = "📊 <b>STATISTICHE PATTERN</b>\n\n"
        
        # Ordina per win rate
        sorted_patterns = sorted(
            data.items(),
            key=lambda x: x[1].get('win_rate', 0),
            reverse=True
        )
        
        for pattern_name, stats in sorted_patterns:
            total_trades = stats['wins'] + stats['losses']
            
            if total_trades == 0:
                # Pattern solo con segnali, no trade completati
                msg += f"🔔 <b>{pattern_name}</b>\n"
                msg += f"  Segnali: {stats['total_signals']}\n"
                msg += f"  Trade: 0 (nessun completato)\n\n"
                continue
            
            win_rate = stats['win_rate']
            avg_pnl = stats['avg_pnl']
            
            # Emoji in base a win rate
            if win_rate >= 70:
                emoji = "🌟"
            elif win_rate >= 60:
                emoji = "✅"
            elif win_rate >= 50:
                emoji = "⚠️"
            else:
                emoji = "❌"
            
            msg += f"{emoji} <b>{pattern_name}</b>\n"
            msg += f"  Segnali: {stats['total_signals']} | Trade: {total_trades}\n"
            msg += f"  Win Rate: <b>{win_rate:.1f}%</b> ({stats['wins']}W/{stats['losses']}L)\n"
            msg += f"  Total PnL: <b>${stats['total_pnl']:+.2f}</b>\n"
            msg += f"  Avg PnL: ${avg_pnl:+.2f}\n"
            
            if stats['best_trade'] > 0:
                msg += f"  Best: ${stats['best_trade']:+.2f}\n"
            if stats['worst_trade'] < 0:
                msg += f"  Worst: ${stats['worst_trade']:+.2f}\n"
            
            if stats['avg_duration_minutes'] > 0:
                hours = int(stats['avg_duration_minutes'] / 60)
                minutes = int(stats['avg_duration_minutes'] % 60)
                msg += f"  Avg Duration: {hours}h {minutes}m\n"
            
            msg += "\n"
        
        # Summary globale
        total_signals = sum(s['total_signals'] for s in data.values())
        total_trades = sum(s['wins'] + s['losses'] for s in data.values())
        total_wins = sum(s['wins'] for s in data.values())
        total_losses = sum(s['losses'] for s in data.values())
        total_pnl = sum(s['total_pnl'] for s in data.values())
        
        global_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        msg += "━━━━━━━━━━━━━━━━━━━\n"
        msg += "<b>TOTALI</b>\n"
        msg += f"Segnali: {total_signals}\n"
        msg += f"Trade: {total_trades} ({total_wins}W/{total_losses}L)\n"
        msg += f"Win Rate: <b>{global_win_rate:.1f}%</b>\n"
        msg += f"Total PnL: <b>${total_pnl:+.2f}</b>\n"
        
        return msg
        
    except Exception as e:
        logging.error(f'Errore get_pattern_stats_summary: {e}')
        return f"❌ Errore generazione report: {str(e)}"


def get_pattern_stats_detailed(pattern_name: str) -> str:
    """
    Statistiche dettagliate per un pattern specifico
    
    Args:
        pattern_name: Nome del pattern
    
    Returns:
        str: Report dettagliato
    """
    try:
        with PATTERN_STATS_LOCK:
            if pattern_name not in PATTERN_STATS:
                return f"❌ Pattern '{pattern_name}' non trovato nelle statistiche."
            
            stats = PATTERN_STATS[pattern_name]
        
        total_trades = stats['wins'] + stats['losses']
        
        msg = f"📊 <b>{pattern_name}</b>\n\n"
        
        # Stats globali
        msg += "<b>📈 Performance Globale:</b>\n"
        msg += f"Segnali totali: {stats['total_signals']}\n"
        msg += f"Trade eseguiti: {stats['total_trades']}\n"
        msg += f"Trade completati: {total_trades}\n"
        
        if total_trades > 0:
            msg += f"Win Rate: <b>{stats['win_rate']:.1f}%</b>\n"
            msg += f"Wins: {stats['wins']} | Losses: {stats['losses']}\n"
            msg += f"Total PnL: <b>${stats['total_pnl']:+.2f}</b>\n"
            msg += f"Avg PnL: ${stats['avg_pnl']:+.2f}\n"
            msg += f"Best Trade: ${stats['best_trade']:+.2f}\n"
            msg += f"Worst Trade: ${stats['worst_trade']:+.2f}\n"
            
            if stats['avg_duration_minutes'] > 0:
                hours = int(stats['avg_duration_minutes'] / 60)
                minutes = int(stats['avg_duration_minutes'] % 60)
                msg += f"Avg Duration: {hours}h {minutes}m\n"
        else:
            msg += "Nessun trade completato ancora.\n"
        
        # Stats per timeframe
        if stats['by_timeframe']:
            msg += "\n<b>📊 Per Timeframe:</b>\n"
            
            for tf in sorted(stats['by_timeframe'].keys()):
                tf_stats = stats['by_timeframe'][tf]
                tf_trades = tf_stats['wins'] + tf_stats['losses']
                
                if tf_trades > 0:
                    tf_wr = (tf_stats['wins'] / tf_trades) * 100
                    msg += f"  • {tf}: {tf_wr:.1f}% WR ({tf_stats['wins']}W/{tf_stats['losses']}L) "
                    msg += f"${tf_stats['pnl']:+.2f}\n"
                else:
                    msg += f"  • {tf}: {tf_stats['signals']} segnali (no trade)\n"
        
        # Top 5 symbols
        if stats['by_symbol']:
            msg += "\n<b>🪙 Top Symbols:</b>\n"
            
            sorted_symbols = sorted(
                stats['by_symbol'].items(),
                key=lambda x: x[1]['pnl'],
                reverse=True
            )[:5]
            
            for symbol, sym_stats in sorted_symbols:
                sym_trades = sym_stats['wins'] + sym_stats['losses']
                
                if sym_trades > 0:
                    sym_wr = (sym_stats['wins'] / sym_trades) * 100
                    msg += f"  • {symbol}: {sym_wr:.1f}% WR "
                    msg += f"${sym_stats['pnl']:+.2f}\n"
        
        # Last updated
        if stats['last_updated']:
            try:
                last_update = datetime.fromisoformat(stats['last_updated'])
                time_str = last_update.strftime('%d/%m %H:%M')
                msg += f"\n⏰ Ultimo aggiornamento: {time_str} UTC"
            except:
                pass
        
        return msg
        
    except Exception as e:
        logging.error(f'Errore get_pattern_stats_detailed: {e}')
        return f"❌ Errore: {str(e)}"


# ----------------------------- INTEGRATION HELPERS -----------------------------

def integrate_pattern_stats_on_signal(
    pattern_name: str,
    symbol: str,
    timeframe: str,
    side: str,
    autotrade: bool
):
    """
    Chiama questa funzione quando rilevi un pattern
    """
    record_pattern_signal(
        pattern_name=pattern_name,
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        traded=autotrade
    )


def integrate_pattern_stats_on_close(
    symbol: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    open_timestamp: str,
    close_timestamp: str
):
    """
    Chiama questa quando una posizione si chiude
    
    Nota: Devi recuperare il pattern_name dal tracking posizione
    """
    try:
        # Calcola durata
        open_time = datetime.fromisoformat(open_timestamp)
        close_time = datetime.fromisoformat(close_timestamp)
        duration = (close_time - open_time).total_seconds() / 60  # minuti
        
        # Recupera pattern dal tracking (se salvato)
        # Per ora assumiamo di averlo nel tracking posizione
        with POSITIONS_LOCK:
            if symbol in ACTIVE_POSITIONS:
                pos_info = ACTIVE_POSITIONS[symbol]
                pattern_name = pos_info.get('pattern_name')
                timeframe = pos_info.get('timeframe')
                side = pos_info.get('side')
                
                if pattern_name:
                    record_pattern_trade_result(
                        pattern_name=pattern_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        side=side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        duration_minutes=int(duration)
                    )
    except Exception as e:
        logging.error(f'Errore integrate_pattern_stats_on_close: {e}')


# ----------------------------- TELEGRAM COMMANDS -----------------------------

async def cmd_pattern_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /pattern_stats - Mostra statistiche tutti i pattern
    /pattern_stats NOME - Statistiche dettagliate pattern specifico
    """
    args = context.args
    
    if not args:
        # Report generale
        msg = get_pattern_stats_summary()
        await update.message.reply_text(msg, parse_mode='HTML')
    else:
        # Report specifico
        pattern_name = ' '.join(args)
        msg = get_pattern_stats_detailed(pattern_name)
        await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_reset_pattern_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /reset_pattern_stats - Reset TUTTE le statistiche (richiede conferma)
    /reset_pattern_stats NOME - Reset statistiche pattern specifico
    """
    args = context.args
    
    if not args:
        # Reset totale (richiede conferma)
        await update.message.reply_text(
            "⚠️ <b>ATTENZIONE</b>\n\n"
            "Stai per cancellare TUTTE le statistiche pattern.\n"
            "Questa azione è irreversibile.\n\n"
            "Per confermare: /reset_pattern_stats CONFIRM",
            parse_mode='HTML'
        )
        return
    
    if args[0] == 'CONFIRM':
        # Reset globale
        with PATTERN_STATS_LOCK:
            PATTERN_STATS.clear()
        
        save_pattern_stats()
        
        await update.message.reply_text(
            "✅ Tutte le statistiche sono state cancellate.",
            parse_mode='HTML'
        )
    else:
        # Reset pattern specifico
        pattern_name = ' '.join(args)
        
        with PATTERN_STATS_LOCK:
            if pattern_name in PATTERN_STATS:
                del PATTERN_STATS[pattern_name]
                save_pattern_stats()
                
                await update.message.reply_text(
                    f"✅ Statistiche di '{pattern_name}' cancellate.",
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text(
                    f"❌ Pattern '{pattern_name}' non trovato.",
                    parse_mode='HTML'
                )


async def cmd_export_pattern_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /export_pattern_stats - Esporta statistiche in formato JSON
    """
    try:
        with PATTERN_STATS_LOCK:
            data = dict(PATTERN_STATS)
        
        # Crea file JSON
        json_str = json.dumps(data, indent=2)
        json_bytes = json_str.encode('utf-8')
        
        # Invia come file
        await update.message.reply_document(
            document=json_bytes,
            filename=f"pattern_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            caption="📊 Export statistiche pattern"
        )
        
    except Exception as e:
        logging.error(f'Errore export_pattern_stats: {e}')
        await update.message.reply_text(f"❌ Errore export: {str(e)}")

"""
Notifications Module - Centralized Telegram Message Management
Elimina codice duplicato per caption e chart generation
"""

import io
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf

import config


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class PatternSignal:
    """Dati completi per un segnale pattern"""
    symbol: str
    timeframe: str
    pattern_name: str
    side: str
    
    # Prezzi
    entry_price: float
    sl_price: float
    tp_price: float
    current_price: float
    
    # Pattern data
    pattern_data: Optional[Dict] = None
    
    # EMA analysis
    ema_analysis: Optional[Dict] = None
    
    # Metriche
    volume_ratio: float = 0.0
    qty: float = 0.0
    risk_usd: float = 0.0
    
    # Flags
    position_exists: bool = False
    autotrade_enabled: bool = False
    
    # Timestamp
    timestamp: Optional[datetime] = None
    
    @property
    def risk_reward(self) -> float:
        """Calcola Risk:Reward ratio"""
        if abs(self.sl_price - self.entry_price) > 0:
            return abs(self.tp_price - self.entry_price) / abs(self.sl_price - self.entry_price)
        return 0.0
    
    @property
    def price_decimals(self) -> int:
        """Calcola decimali dinamici per prezzi"""
        return get_price_decimals(self.current_price)


@dataclass
class MarketSnapshot:
    """Snapshot mercato senza pattern"""
    symbol: str
    timeframe: str
    current_price: float
    ema_analysis: Optional[Dict] = None
    timestamp: Optional[datetime] = None
    
    @property
    def price_decimals(self) -> int:
        return get_price_decimals(self.current_price)


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_price_decimals(price: float) -> int:
    """Determina decimali in base al prezzo"""
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


def format_ema_section(ema_analysis: Dict) -> str:
    """
    Formatta sezione EMA Analysis per caption
    
    Returns:
        Stringa HTML formattata
    """
    if not ema_analysis:
        return ""
    
    quality_emoji_map = {
        'GOLD': '🌟',
        'GOOD': '✅',
        'OK': '⚠️',
        'WEAK': '🔶',
        'BAD': '❌'
    }
    
    q_emoji = quality_emoji_map.get(ema_analysis['quality'], '⚪')
    
    section = "\n━━━━━━━━━━━━━━━━━━━\n"
    section += "📈 <b>EMA Analysis</b>\n\n"
    section += f"{q_emoji} Quality: <b>{ema_analysis['quality']}</b>\n"
    section += f"Score: <b>{ema_analysis['score']}/100</b>\n\n"
    section += ema_analysis['details']
    
    # Valori EMA
    if 'ema_values' in ema_analysis:
        ema_vals = ema_analysis['ema_values']
        price_decimals = get_price_decimals(ema_vals['price'])
        
        section += "\n\n💡 <b>EMA Values:</b>\n"
        section += f"Price: ${ema_vals['price']:.{price_decimals}f}\n"
        section += f"EMA 5: ${ema_vals['ema5']:.{price_decimals}f}\n"
        section += f"EMA 10: ${ema_vals['ema10']:.{price_decimals}f}\n"
        section += f"EMA 60: ${ema_vals['ema60']:.{price_decimals}f}\n"
        section += f"EMA 223: ${ema_vals['ema223']:.{price_decimals}f}\n"
    
    return section


def format_risk_section(signal: PatternSignal) -> str:
    """Formatta sezione risk management"""
    section = "\n📊 <b>Risk Management:</b>\n"
    section += f"Position Size: {signal.qty:.4f}\n"
    section += f"Risk per Trade: ${signal.risk_usd:.2f}\n"
    
    if signal.ema_analysis:
        score = signal.ema_analysis['score']
        quality = signal.ema_analysis['quality']
        
        section += f"EMA Score: {score}/100 ({quality})\n"
        section += "Risk Tier: "
        
        if score >= 80:
            section += "🌟 GOLD (Max Risk)\n"
        elif score >= 60:
            section += "✅ GOOD (Standard Risk)\n"
        elif score >= 40:
            section += "⚠️ OK (Reduced Risk)\n"
        else:
            section += "❌ WEAK (Minimal Risk)\n"
    
    return section


def format_filters_section() -> str:
    """Formatta info filtri configurati"""
    section = "\n💡 <b>Filtri Pattern:</b>\n"
    
    if config.TREND_FILTER_ENABLED:
        section += f"Trend: {config.TREND_FILTER_MODE.upper()}"
        if config.TREND_FILTER_MODE == 'ema_based':
            section += " (Price > EMA 60)\n"
        elif config.TREND_FILTER_MODE == 'structure':
            section += " (HH+HL)\n"
        else:
            section += "\n"
    else:
        section += "Trend: OFF\n"
    
    if config.VOLUME_FILTER_ENABLED:
        section += f"Volume: {config.VOLUME_FILTER_MODE.upper()}\n"
    else:
        section += "Volume: OFF\n"
    
    if config.EMA_FILTER_ENABLED:
        section += f"EMA: {config.EMA_FILTER_MODE.upper()}\n"
    else:
        section += "EMA: OFF\n"
    
    return section


# ============================================
# PATTERN-SPECIFIC CAPTIONS
# ============================================

class PatternCaptionBuilder:
    """Builder per caption pattern-specific"""
    
    @staticmethod
    def build_generic(signal: PatternSignal) -> str:
        """Caption generica per pattern standard"""
        pd = signal.price_decimals
        
        caption = "🔥 <b>SEGNALE BUY</b>\n\n"
        
        # EMA Quality
        if signal.ema_analysis:
            q_emoji = {
                'GOLD': '🌟', 'GOOD': '✅', 'OK': '⚠️',
                'WEAK': '🔶', 'BAD': '❌'
            }.get(signal.ema_analysis['quality'], '⚪')
            
            caption += f"{q_emoji} EMA Quality: <b>{signal.ema_analysis['quality']}</b>\n"
            caption += f"Score: <b>{signal.ema_analysis['score']}/100</b>\n\n"
        
        # Pattern info
        caption += f"📊 Pattern: <b>{signal.pattern_name}</b>\n"
        caption += f"🪙 Symbol: <b>{signal.symbol}</b> ({signal.timeframe})\n"
        
        if signal.timestamp:
            caption += f"🕐 {signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        
        # Trading params
        caption += f"💵 Entry: <b>${signal.entry_price:.{pd}f}</b>\n"
        caption += f"🛑 Stop Loss: <b>${signal.sl_price:.{pd}f}</b>\n"
        caption += f"🎯 Take Profit: <b>${signal.tp_price:.{pd}f}</b>\n"
        caption += f"📦 Qty: <b>{signal.qty:.4f}</b>\n"
        caption += f"💰 Risk: <b>${signal.risk_usd:.2f}</b>\n"
        caption += f"📏 R:R: <b>{signal.risk_reward:.2f}:1</b>\n"
        
        # EMA Analysis
        if signal.ema_analysis:
            caption += format_ema_section(signal.ema_analysis)
        
        # Risk section
        caption += format_risk_section(signal)
        
        # Filters
        caption += format_filters_section()
        
        # Position warning
        if signal.position_exists:
            caption += "\n\n🚫 <b>Posizione già aperta</b>"
            caption += f"\nOrdine NON eseguito per {signal.symbol}"
        
        # Autotrade status
        if signal.autotrade_enabled and signal.qty > 0 and not signal.position_exists:
            caption += f"\n\n✅ <b>Ordine su Bybit {config.TRADING_MODE.upper()}</b>"
        
        return caption
    
    @staticmethod
    def build_triple_touch(signal: PatternSignal) -> str:
        """Caption specifica per Triple Touch Breakout"""
        pd = signal.price_decimals
        data = signal.pattern_data or {}
        
        caption = f"🎯 <b>TRIPLE TOUCH BREAKOUT</b> ({data.get('quality', 'N/A')})\n\n"
        
        # Pattern specifics
        caption += f"📍 Resistance: ${data.get('resistance', 0):.{pd}f}\n"
        caption += f"🔄 Rejections: {data.get('touch_1_rejection_pct', 0):.1f}% + {data.get('touch_2_rejection_pct', 0):.1f}%\n"
        caption += f"📊 Consolidation: {data.get('consolidation_duration', 0)} candele ({data.get('range_pct', 0):.2f}%)\n\n"
        
        caption += f"💵 Entry: <b>${signal.entry_price:.{pd}f}</b>\n"
        caption += f"🛑 Stop Loss: <b>${signal.sl_price:.{pd}f}</b>\n"
        caption += f"   (sotto consolidamento)\n"
        caption += f"🎯 Take Profit: <b>${signal.tp_price:.{pd}f}</b>\n"
        caption += f"   (2.5R projection)\n\n"
        
        caption += "📊 <b>Quality Metrics:</b>\n"
        caption += f"• Breakout volume: {data.get('volume_ratio', 0):.1f}x\n"
        caption += f"• Breakout body: {data.get('breakout_body_pct', 0):.1f}%\n"
        caption += f"• EMA 60: ${data.get('ema60', 0):.{pd}f}\n"
        caption += f"• EMA aligned: {'✅' if data.get('ema_aligned') else '⚠️'}\n"
        
        caption += format_risk_section(signal)
        
        return caption
    
    @staticmethod
    def build_bullish_flag(signal: PatternSignal) -> str:
        """Caption specifica per Bullish Flag Breakout"""
        pd = signal.price_decimals
        data = signal.pattern_data or {}
        
        caption = f"🚩 <b>BULLISH FLAG BREAKOUT</b>\n\n"
        
        caption += f"💥 Breakout Level X: <b>${data.get('X', 0):.{pd}f}</b>\n"
        caption += f"📏 Pole Height: {data.get('pole_height_pct', 0):.2f}%\n\n"
        
        caption += f"💵 Entry: <b>${signal.entry_price:.{pd}f}</b>\n"
        caption += f"🛑 Stop Loss: <b>${signal.sl_price:.{pd}f}</b>\n"
        caption += f"🎯 Take Profit: <b>${signal.tp_price:.{pd}f}</b>\n\n"
        
        caption += "📊 <b>Pattern Quality:</b>\n"
        caption += f"• Volume: {data.get('volume_ratio', 0):.1f}x\n"
        caption += f"• Flag Duration: {data.get('flag_duration', 0)} candele\n"
        
        caption += format_risk_section(signal)
        
        return caption
    
    @staticmethod
    def build_bud_pattern(signal: PatternSignal) -> str:
        """Caption specifica per BUD/MAXI BUD Pattern"""
        pd = signal.price_decimals
        data = signal.pattern_data or {}
        
        pattern_type = data.get('pattern_type', 'BUD')
        tier = 'MAXI' if 'MAXI' in pattern_type else 'STANDARD'
        
        caption = f"🌱 <b>{pattern_type.upper()}</b>\n\n"
        
        if tier == 'MAXI':
            caption += f"⭐ <b>Setup PREMIUM</b> ({data.get('rest_count', 0)} candele riposo)\n\n"
        else:
            caption += f"📊 <b>Setup VALIDO</b> ({data.get('rest_count', 0)} candele riposo)\n\n"
        
        caption += "💥 <b>Breakout Phase:</b>\n"
        caption += f"  High: ${data.get('breakout_high', 0):.{pd}f}\n"
        caption += f"  Low: ${data.get('breakout_low', 0):.{pd}f}\n"
        caption += f"  Range: ${data.get('breakout_range', 0):.{pd}f}\n"
        caption += f"  Body: {data.get('breakout_body_pct', 0):.1f}%\n\n"
        
        caption += "🛌 <b>Rest Phase:</b>\n"
        caption += f"  Candele: {data.get('rest_count', 0)}\n"
        caption += f"  Avg Range: {data.get('rest_range_pct', 0):.1f}% del breakout\n\n"
        
        caption += "🎯 <b>Trade Setup:</b>\n"
        caption += f"  Entry: ${signal.entry_price:.{pd}f}\n"
        caption += f"  SL: ${signal.sl_price:.{pd}f} (sotto breakout low)\n"
        caption += f"  TP: ${signal.tp_price:.{pd}f} (2R)\n\n"
        
        caption += format_risk_section(signal)
        
        caption += "\n💡 <b>Strategy Notes:</b>\n"
        caption += "  • Breakout + riposo = buyers confidenti\n"
        caption += "  • Pattern compresso = energia per pump\n"
        
        if tier == 'MAXI':
            caption += "  • ⭐ MAXI: 3+ riposo = setup superiore\n"
        
        return caption
    
    @staticmethod
    def build(signal: PatternSignal) -> str:
        """
        Factory method: seleziona builder appropriato
        """
        pattern_builders = {
            'Triple Touch Breakout': PatternCaptionBuilder.build_triple_touch,
            'Bullish Flag Breakout': PatternCaptionBuilder.build_bullish_flag,
            'BUD Pattern': PatternCaptionBuilder.build_bud_pattern,
            'MAXI BUD Pattern': PatternCaptionBuilder.build_bud_pattern,
        }
        
        builder = pattern_builders.get(
            signal.pattern_name,
            PatternCaptionBuilder.build_generic
        )
        
        return builder(signal)


# ============================================
# CHART GENERATION
# ============================================

class ChartGenerator:
    """Genera grafici candlestick con EMA"""
    
    @staticmethod
    def generate(df: pd.DataFrame, symbol: str, timeframe: str) -> io.BytesIO:
        """
        Genera grafico candlestick con EMA overlay
        
        Args:
            df: DataFrame OHLCV
            symbol: Symbol (es. BTCUSDT)
            timeframe: Timeframe (es. 15m)
        
        Returns:
            BytesIO buffer con immagine PNG
        """
        try:
            # Usa ultimi 100 candles
            chart_df = df.tail(100).copy()
            
            # Calcola EMA
            ema_5 = chart_df['close'].ewm(span=5, adjust=False).mean()
            ema_10 = chart_df['close'].ewm(span=10, adjust=False).mean()
            ema_60 = chart_df['close'].ewm(span=60, adjust=False).mean()
            ema_223 = chart_df['close'].ewm(span=223, adjust=False).mean()
            
            # Crea plot addizionali per EMA
            apds = [
                mpf.make_addplot(ema_5, color='#00FF00', width=1.5, label='EMA 5'),
                mpf.make_addplot(ema_10, color='#FFA500', width=1.5, label='EMA 10'),
                mpf.make_addplot(ema_60, color='#FF1493', width=1.5, label='EMA 60'),
                mpf.make_addplot(ema_223, color='#1E90FF', width=2, label='EMA 223')
            ]
            
            # Buffer per salvare
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
            
            # Genera plot
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
            logging.error(f'Errore generazione grafico: {e}')
            raise


# ============================================
# NOTIFICATION MANAGER
# ============================================

class NotificationManager:
    """
    Manager centralizzato per notifiche Telegram
    Elimina codice duplicato in analyze_job
    """
    
    def __init__(self):
        self.chart_generator = ChartGenerator()
        self.caption_builder = PatternCaptionBuilder()
    
    async def send_pattern_signal(
        self,
        context,
        chat_id: int,
        signal: PatternSignal,
        df: pd.DataFrame
    ):
        """
        Invia notifica completa per pattern rilevato
        
        Args:
            context: Telegram context
            chat_id: Chat ID destinazione
            signal: Dati segnale pattern
            df: DataFrame per grafico
        """
        try:
            # 1. Genera caption
            caption = self.caption_builder.build(signal)
            
            # 2. Genera grafico
            chart_buffer = self.chart_generator.generate(
                df,
                signal.symbol,
                signal.timeframe
            )
            
            # 3. Invia
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=chart_buffer,
                caption=caption,
                parse_mode='HTML'
            )
            
            logging.info(
                f"📸 Sent pattern signal: {signal.symbol} {signal.timeframe} - "
                f"{signal.pattern_name}"
            )
            
        except Exception as e:
            logging.error(f'Errore invio pattern signal: {e}')
            
            # Fallback: invia solo testo
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"⚠️ Errore grafico\n\n{caption}",
                    parse_mode='HTML'
                )
            except:
                logging.error('Fallback send failed')
    
    async def send_market_snapshot(
        self,
        context,
        chat_id: int,
        snapshot: MarketSnapshot,
        df: pd.DataFrame
    ):
        """
        Invia snapshot mercato (senza pattern, solo EMA analysis)
        
        Args:
            context: Telegram context
            chat_id: Chat ID destinazione
            snapshot: Dati snapshot
            df: DataFrame per grafico
        """
        try:
            pd = snapshot.price_decimals
            
            # Caption
            caption = f"📊 <b>{snapshot.symbol}</b> ({snapshot.timeframe})\n"
            
            if snapshot.timestamp:
                caption += f"🕐 {snapshot.timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n"
            
            caption += f"💵 Price: ${snapshot.current_price:.{pd}f}\n\n"
            caption += "🔔 <b>Full Mode - Nessun pattern rilevato</b>\n\n"
            
            # Info filtri
            caption += "💡 <b>Filter Configuration:</b>\n"
            caption += format_filters_section()
            
            # EMA Analysis
            if snapshot.ema_analysis:
                caption += format_ema_section(snapshot.ema_analysis)
                
                # Suggerimenti
                quality = snapshot.ema_analysis['quality']
                caption += "\n\n💡 <b>Suggerimento:</b>\n"
                
                if quality == 'GOLD':
                    caption += "🌟 Setup PERFETTO! Aspetta pattern qui."
                elif quality == 'GOOD':
                    caption += "✅ Buone condizioni. Setup valido."
                elif quality == 'OK':
                    caption += "⚠️ Accettabile ma non ottimale."
                elif quality == 'WEAK':
                    caption += "🔶 Condizioni deboli. Meglio aspettare."
                else:
                    caption += "❌ Condizioni sfavorevoli. NO entry."
            
            # Grafico
            chart_buffer = self.chart_generator.generate(
                df,
                snapshot.symbol,
                snapshot.timeframe
            )
            
            # Invia
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=chart_buffer,
                caption=caption,
                parse_mode='HTML'
            )
            
            logging.info(
                f"📸 Sent market snapshot: {snapshot.symbol} {snapshot.timeframe}"
            )
            
        except Exception as e:
            logging.error(f'Errore invio market snapshot: {e}')


# ============================================
# GLOBAL INSTANCE
# ============================================

NOTIFICATION_MANAGER = NotificationManager()


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

async def send_pattern_notification(
    context,
    chat_id: int,
    symbol: str,
    timeframe: str,
    pattern_name: str,
    side: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    current_price: float,
    qty: float,
    risk_usd: float,
    df: pd.DataFrame,
    pattern_data: Optional[Dict] = None,
    ema_analysis: Optional[Dict] = None,
    position_exists: bool = False,
    autotrade_enabled: bool = False,
    timestamp: Optional[datetime] = None
):
    """
    Convenience function per inviare notifica pattern
    Sostituisce il blocco di 100+ righe in analyze_job
    """
    signal = PatternSignal(
        symbol=symbol,
        timeframe=timeframe,
        pattern_name=pattern_name,
        side=side,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        current_price=current_price,
        pattern_data=pattern_data,
        ema_analysis=ema_analysis,
        qty=qty,
        risk_usd=risk_usd,
        position_exists=position_exists,
        autotrade_enabled=autotrade_enabled,
        timestamp=timestamp or datetime.now(timezone.utc)
    )
    
    await NOTIFICATION_MANAGER.send_pattern_signal(
        context,
        chat_id,
        signal,
        df
    )


async def send_market_notification(
    context,
    chat_id: int,
    symbol: str,
    timeframe: str,
    current_price: float,
    df: pd.DataFrame,
    ema_analysis: Optional[Dict] = None,
    timestamp: Optional[datetime] = None
):
    """
    Convenience function per inviare snapshot mercato
    """
    snapshot = MarketSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        ema_analysis=ema_analysis,
        timestamp=timestamp or datetime.now(timezone.utc)
    )
    
    await NOTIFICATION_MANAGER.send_market_snapshot(
        context,
        chat_id,
        snapshot,
        df
    )

"""
Break-Even Manager - Sistema Multi-Layer per limitare perdite
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple
import config

class BreakEvenManager:
    """Gestisce break-even automatico con multiple strategie"""
    
    def __init__(self):
        self.last_checks = {}  # symbol -> last_check_time
    
    def should_activate_breakeven(
        self,
        symbol: str,
        pos_info: dict,
        current_price: float,
        df_current: 'pd.DataFrame'
    ) -> Tuple[bool, Optional[float], str]:
        """
        Determina se attivare break-even e calcola nuovo SL
        
        Returns:
            (should_activate, new_sl, reason)
        """
        if not config.BREAKEVEN_ENABLED:
            return (False, None, "Break-even disabled")
        
        entry_price = pos_info.get('entry_price')
        side = pos_info.get('side')
        timestamp_str = pos_info.get('timestamp')
        timeframe = pos_info.get('timeframe', '15m')
        current_sl = pos_info.get('sl', 0)
        
        if not all([entry_price, side, timestamp_str]):
            return (False, None, "Missing position data")
        
        # Parse timestamp
        try:
            entry_time = datetime.fromisoformat(timestamp_str)
        except:
            return (False, None, "Invalid timestamp")
        
        now = datetime.now(timezone.utc)
        time_elapsed = (now - entry_time).total_seconds() / 60  # minuti
        
        # Calcola profit corrente
        if side == 'Buy':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
        
        # ===== METODO 1: TIME-BASED =====
        if config.BREAKEVEN_CONFIG['time_based']['enabled']:
            trigger_minutes = config.BREAKEVEN_CONFIG['time_based']['minutes']
            
            if time_elapsed >= trigger_minutes:
                # Sposta a break-even (entry + buffer)
                buffer = config.BREAKEVEN_CONFIG['time_based']['buffer_pct']
                
                if side == 'Buy':
                    new_sl = entry_price * (1 + buffer)
                    
                    # Solo se migliora lo SL corrente
                    if new_sl > current_sl:
                        reason = f"Time-based: {time_elapsed:.1f}min >= {trigger_minutes}min"
                        return (True, new_sl, reason)
                else:
                    new_sl = entry_price * (1 - buffer)
                    
                    if new_sl < current_sl:
                        reason = f"Time-based: {time_elapsed:.1f}min >= {trigger_minutes}min"
                        return (True, new_sl, reason)
        
        # ===== METODO 2: CANDLE-BASED =====
        if config.BREAKEVEN_CONFIG['candle_based']['enabled']:
            min_candles = config.BREAKEVEN_CONFIG['candle_based']['min_green_candles']
            
            # Conta candele verdi consecutive (ultime N)
            if len(df_current) >= min_candles + 1:
                last_candles = df_current.tail(min_candles + 1)
                
                # Escludi candela corrente (non ancora chiusa)
                closed_candles = last_candles.iloc[:-1]
                
                if side == 'Buy':
                    # Conta quante sono verdi
                    green_count = (closed_candles['close'] > closed_candles['open']).sum()
                    
                    if green_count >= min_candles:
                        buffer = config.BREAKEVEN_CONFIG['candle_based']['buffer_pct']
                        new_sl = entry_price * (1 + buffer)
                        
                        if new_sl > current_sl:
                            reason = f"Candle-based: {green_count} green candles"
                            return (True, new_sl, reason)
                else:
                    # Per SHORT: conta candele rosse
                    red_count = (closed_candles['close'] < closed_candles['open']).sum()
                    
                    if red_count >= min_candles:
                        buffer = config.BREAKEVEN_CONFIG['candle_based']['buffer_pct']
                        new_sl = entry_price * (1 - buffer)
                        
                        if new_sl < current_sl:
                            reason = f"Candle-based: {red_count} red candles"
                            return (True, new_sl, reason)
        
        # ===== METODO 3: PROFIT-BASED =====
        if config.BREAKEVEN_CONFIG['profit_based']['enabled']:
            min_profit = config.BREAKEVEN_CONFIG['profit_based']['min_profit_pct']
            
            if profit_pct >= min_profit:
                lock_pct = config.BREAKEVEN_CONFIG['profit_based']['lock_pct']
                
                if side == 'Buy':
                    new_sl = entry_price * (1 + lock_pct / 100)
                    
                    if new_sl > current_sl:
                        reason = f"Profit-based: {profit_pct:.2f}% >= {min_profit}%"
                        return (True, new_sl, reason)
                else:
                    new_sl = entry_price * (1 - lock_pct / 100)
                    
                    if new_sl < current_sl:
                        reason = f"Profit-based: {profit_pct:.2f}% >= {min_profit}%"
                        return (True, new_sl, reason)
        
        # ===== METODO 4: QUICK EXIT (se in perdita dopo N min) =====
        if config.BREAKEVEN_CONFIG['quick_exit']['enabled']:
            check_after = config.BREAKEVEN_CONFIG['quick_exit']['check_after_minutes']
            max_loss = config.BREAKEVEN_CONFIG['quick_exit']['max_loss_pct']
            
            if time_elapsed >= check_after:
                if profit_pct < max_loss:
                    # Setup è fallito, esci con perdita controllata
                    # Questo NON sposta lo SL, ma segnala di chiudere
                    reason = f"Quick Exit: {profit_pct:.2f}% < {max_loss}% after {time_elapsed:.1f}min"
                    
                    # Restituisci entry come "SL" (= chiudi a mercato)
                    return (True, entry_price, reason)
        
        return (False, None, "No break-even condition met")
    
    def should_quick_exit(
        self,
        symbol: str,
        pos_info: dict,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Verifica se eseguire quick exit (chiusura immediata)
        
        Returns:
            (should_exit, reason)
        """
        if not config.BREAKEVEN_CONFIG['quick_exit']['enabled']:
            return (False, "Quick exit disabled")
        
        entry_price = pos_info.get('entry_price')
        side = pos_info.get('side')
        timestamp_str = pos_info.get('timestamp')
        
        if not all([entry_price, side, timestamp_str]):
            return (False, "Missing data")
        
        try:
            entry_time = datetime.fromisoformat(timestamp_str)
        except:
            return (False, "Invalid timestamp")
        
        now = datetime.now(timezone.utc)
        time_elapsed = (now - entry_time).total_seconds() / 60
        
        check_after = config.BREAKEVEN_CONFIG['quick_exit']['check_after_minutes']
        max_loss = config.BREAKEVEN_CONFIG['quick_exit']['max_loss_pct']
        
        if time_elapsed < check_after:
            return (False, f"Too early ({time_elapsed:.1f}min < {check_after}min)")
        
        # Calcola profit
        if side == 'Buy':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
        
        if profit_pct < max_loss:
            reason = (
                f"Quick Exit triggered: Loss {profit_pct:.2f}% "
                f"exceeds max {max_loss}% after {time_elapsed:.1f}min"
            )
            return (True, reason)
        
        return (False, "Within acceptable loss range")

# Global instance
BREAKEVEN_MANAGER = BreakEvenManager()

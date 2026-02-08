"""
Pattern Registry System - Unified Pattern Detection
Sostituisce l'approccio if/elif con un sistema basato su registry

VERSION: 2.0 - Aggiornato con tutti i pattern esistenti + Enhanced variants
"""

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple
import pandas as pd
import logging

@dataclass
class PatternConfig:
    """Configurazione pattern unificata"""
    name: str
    key: str
    detector: Callable  # Funzione che rileva il pattern
    enabled: bool
    side: str  # 'Buy', 'Sell', 'Both'
    tier: int  # 1=High, 2=Good, 3=Classic
    emoji: str
    description: str
    min_volume_ratio: float = 1.5  # Soglia volume custom
    require_ema_filter: bool = True
    require_trend_filter: bool = True
    order_type: str = 'market'  # 'market' o 'limit'


class PatternRegistry:
    """
    Registry centralizzato per tutti i pattern.
    Elimina duplicazione e semplifica aggiunta nuovi pattern.
    """
    
    def __init__(self):
        self.patterns: Dict[str, PatternConfig] = {}
        self._register_all_patterns()
    
    def register(self, config: PatternConfig):
        """Registra un pattern"""
        self.patterns[config.key] = config
        logging.debug(f"Pattern registrato: {config.name}")
    
    def _register_all_patterns(self):
        """Registra tutti i pattern disponibili"""
        
        # ═══════════════════════════════════════════
        # TIER 1 - High Probability (60-72%)
        # ═══════════════════════════════════════════
        
        self.register(PatternConfig(
            name="Volume Spike Breakout",
            key="volume_spike_breakout",
            detector=self._detect_volume_spike,
            enabled=True,
            side="Buy",
            tier=1,
            emoji="📊💥",
            description="Breakout volume 3x+, EMA alignment",
            min_volume_ratio=2.5,
            order_type='market'
        ))
        
        self.register(PatternConfig(
            name="Breakout + Retest",
            key="breakout_retest",
            detector=self._detect_breakout_retest,
            enabled=True,
            side="Buy",
            tier=1,
            emoji="🔄📈",
            description="Consolidation → Breakout → Retest → Bounce",
            min_volume_ratio=2.0,
            order_type='limit'
        ))
        
        self.register(PatternConfig(
            name="Triple Touch Breakout",
            key="triple_touch_breakout",
            detector=self._detect_triple_touch,
            enabled=True,
            side="Buy",
            tier=1,
            emoji="🎯3️⃣",
            description="3 tocchi resistance + breakout sopra EMA 60",
            min_volume_ratio=2.0,
            order_type='market'
        ))
        
        self.register(PatternConfig(
            name="Liquidity Sweep + Reversal",
            key="liquidity_sweep_reversal",
            detector=self._detect_liquidity_sweep,
            enabled=True,
            side="Buy",
            tier=1,
            emoji="💎",
            description="Smart money sweep + reversal",
            min_volume_ratio=2.0,
            order_type='market'
        ))
        
        self.register(PatternConfig(
            name="BUD Pattern",
            key="bud_pattern",
            detector=lambda df: self._detect_bud(df, require_maxi=False),
            enabled=True,
            side="Buy",
            tier=1,
            emoji="🌱",
            description="Breakout + 2 candele riposo nel range",
            min_volume_ratio=1.5,
            order_type='market'
        ))
        
        self.register(PatternConfig(
            name="MAXI BUD Pattern",
            key="maxi_bud_pattern",
            detector=lambda df: self._detect_bud(df, require_maxi=True),
            enabled=True,
            side="Buy",
            tier=1,
            emoji="🌟🌱",
            description="Breakout + 3+ candele riposo (setup più forte)",
            min_volume_ratio=1.5,
            order_type='market'
        ))
        
        # ═══════════════════════════════════════════
        # TIER 2 - Good (52-62%)
        # ═══════════════════════════════════════════
        
        self.register(PatternConfig(
            name="Support/Resistance Bounce",
            key="sr_bounce",
            detector=self._detect_sr_bounce,
            enabled=True,
            side="Buy",
            tier=2,
            emoji="🎯",
            description="Bounce su S/R con rejection",
            min_volume_ratio=1.0,  # Più permissivo per 5m
            order_type='limit'
        ))
        
        self.register(PatternConfig(
            name="Bullish Flag Breakout",
            key="bullish_flag_breakout",
            detector=self._detect_flag,
            enabled=True,
            side="Buy",
            tier=2,
            emoji="🚩",
            description="Pole + flag + breakout (vol 2x+)",
            min_volume_ratio=2.0,
            order_type='limit'
        ))
        
        self.register(PatternConfig(
            name="Bullish Comeback",
            key="bullish_comeback",
            detector=self._detect_comeback,
            enabled=True,
            side="Buy",
            tier=2,
            emoji="🔄",
            description="Inversione dopo tentativo ribassista",
            min_volume_ratio=1.5,
            order_type='market'
        ))
        
        self.register(PatternConfig(
            name="Compression Breakout",
            key="compression_breakout",
            detector=self._detect_compression,
            enabled=True,
            side="Buy",
            tier=2,
            emoji="💥",
            description="EMA compression + breakout (RSI, vol, HTF)",
            min_volume_ratio=1.8,
            order_type='market'
        ))
        
        self.register(PatternConfig(
            name="Morning Star EMA Breakout",
            key="morning_star_ema_breakout",
            detector=lambda df: self._detect_morning_star_ema_breakout(df),
            enabled=True,
            side="Buy",
            tier=2,
            emoji="⭐💥",
            description="Morning Star + EMA breakthrough (IMPROVED - Early Entry)",
            min_volume_ratio=1.3,  # Più permissivo
            order_type='market'
        ))
        
        self.register(PatternConfig(
            name="Higher Low Breakout",
            key="higher_low_breakout",
            detector=self._detect_higher_low,
            enabled=True,
            side="Buy",
            tier=2,
            emoji="📈🔺",
            description="Impulso + higher lows + breakout",
            min_volume_ratio=1.5,
            order_type='limit'
        ))
        
        # ═══════════════════════════════════════════
        # TIER 3 - Enhanced Patterns (Classic + EMA)
        # ═══════════════════════════════════════════
        
        self.register(PatternConfig(
            name="Bullish Engulfing Enhanced",
            key="bullish_engulfing",
            detector=self._detect_engulfing,
            enabled=True,
            side="Buy",
            tier=3,
            emoji="🟢",
            description="Engulfing su EMA (Enhanced)",
            min_volume_ratio=1.8,
            order_type='limit'
        ))
        
        self.register(PatternConfig(
            name="Pin Bar Bullish Enhanced",
            key="pin_bar_bullish",
            detector=self._detect_pin_bar,
            enabled=True,
            side="Buy",
            tier=3,
            emoji="📍",
            description="Pin bar su EMA (Enhanced)",
            min_volume_ratio=1.5,
            order_type='market'
        ))
        
        self.register(PatternConfig(
            name="Morning Star Enhanced",
            key="morning_star",
            detector=self._detect_morning_star,
            enabled=True,
            side="Buy",
            tier=3,
            emoji="⭐",
            description="3 candele reversal su EMA (Enhanced)",
            min_volume_ratio=1.8,
            order_type='limit'
        ))
    
    # ============================================
    # DETECTOR WRAPPERS (collegano alle funzioni esistenti)
    # ============================================
    
    def _detect_volume_spike(self, df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """Wrapper per is_volume_spike_breakout"""
        from bybit_telegram_bot_fixed import is_volume_spike_breakout
        return is_volume_spike_breakout(df)
    
    def _detect_breakout_retest(self, df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """Wrapper per is_breakout_retest"""
        from bybit_telegram_bot_fixed import is_breakout_retest
        return is_breakout_retest(df)
    
    def _detect_triple_touch(self, df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """Wrapper per is_triple_touch_breakout"""
        from bybit_telegram_bot_fixed import is_triple_touch_breakout
        return is_triple_touch_breakout(df)
    
    def _detect_liquidity_sweep(self, df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """Wrapper per is_liquidity_sweep_reversal"""
        from bybit_telegram_bot_fixed import is_liquidity_sweep_reversal
        return is_liquidity_sweep_reversal(df)
    
    def _detect_sr_bounce(self, df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """Wrapper per is_support_resistance_bounce"""
        from bybit_telegram_bot_fixed import is_support_resistance_bounce
        return is_support_resistance_bounce(df)
    
    def _detect_flag(self, df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """Wrapper per is_bullish_flag_breakout"""
        from bybit_telegram_bot_fixed import is_bullish_flag_breakout
        return is_bullish_flag_breakout(df)
    
    def _detect_bud(self, df: pd.DataFrame, require_maxi: bool = False) -> Tuple[bool, Optional[Dict]]:
        """Wrapper per is_bud_pattern"""
        from bybit_telegram_bot_fixed import is_bud_pattern
        return is_bud_pattern(df, require_maxi=require_maxi)
    
    def _detect_comeback(self, df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """
        Wrapper per is_bullish_comeback
        Nota: ritorna solo bool, quindi creiamo dict vuoto
        """
        from bybit_telegram_bot_fixed import is_bullish_comeback
        found = is_bullish_comeback(df)
        return (found, {} if found else None)
    
    def _detect_compression(self, df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """
        Wrapper per is_compression_breakout
        Nota: ritorna solo bool, quindi creiamo dict vuoto
        """
        from bybit_telegram_bot_fixed import is_compression_breakout
        found = is_compression_breakout(df)
        return (found, {} if found else None)
    
    def _detect_morning_star_ema_breakout(self, df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """Wrapper per Morning Star EMA Breakout IMPROVED"""
        from bybit_telegram_bot_fixed import is_morning_star_ema_breakout_improved
        
        result = is_morning_star_ema_breakout_improved(df)
        
        if isinstance(result, tuple) and len(result) == 2:
            return result
        else:
            return (False, None)
    
    def _detect_higher_low(self, df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """Wrapper per is_higher_low_consolidation_breakout"""
        from bybit_telegram_bot_fixed import is_higher_low_consolidation_breakout
        return is_higher_low_consolidation_breakout(df)
    
    def _detect_engulfing(self, df: pd.DataFrame) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Wrapper per is_bullish_engulfing_enhanced"""
        from bybit_telegram_bot_fixed import is_bullish_engulfing_enhanced
        if len(df) < 2:
            return (False, None, None)
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        return is_bullish_engulfing_enhanced(prev, curr, df)
    
    def _detect_pin_bar(self, df: pd.DataFrame) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Wrapper per is_pin_bar_bullish_enhanced"""
        from bybit_telegram_bot_fixed import is_pin_bar_bullish_enhanced
        if len(df) < 1:
            return (False, None, None)
        curr = df.iloc[-1]
        return is_pin_bar_bullish_enhanced(curr, df)
    
    def _detect_morning_star(self, df: pd.DataFrame) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Wrapper per is_morning_star_enhanced"""
        from bybit_telegram_bot_fixed import is_morning_star_enhanced
        return is_morning_star_enhanced(df)
    
    # ============================================
    # METODI PUBBLICI
    # ============================================
    
    def get_enabled_patterns(self, tier: Optional[int] = None) -> Dict[str, PatternConfig]:
        """Ottiene pattern abilitati, opzionalmente filtrati per tier"""
        patterns = {k: v for k, v in self.patterns.items() if v.enabled}
        
        if tier is not None:
            patterns = {k: v for k, v in patterns.items() if v.tier == tier}
        
        return patterns
    
    def detect_all(self, df: pd.DataFrame, symbol: str = None) -> Tuple[bool, Optional[str], Optional[str], Optional[Dict]]:
        """
        Testa TUTTI i pattern abilitati in ordine di priorità (tier).
        Ritorna il PRIMO pattern trovato.
        
        Returns:
            (found, side, pattern_name, pattern_data)
        """
        # Ordina per tier (1 = priority massima)
        sorted_patterns = sorted(
            self.get_enabled_patterns().values(),
            key=lambda p: p.tier
        )
        
        for config in sorted_patterns:
            try:
                logging.debug(f"Testing pattern: {config.name}")
                
                # Chiama detector
                result = config.detector(df)
                
                # Handle diversi tipi di return
                if len(result) == 2:
                    # (bool, data) - pattern normali
                    found, data = result
                    tier = None
                elif len(result) == 3:
                    # (bool, tier, data) - enhanced patterns
                    found, tier, data = result
                else:
                    logging.warning(f"Pattern {config.name} returned invalid format")
                    continue
                
                if found:
                    logging.info(f"✅ Pattern FOUND: {config.name}")
                    
                    # Arricchisci data con info pattern
                    if data is None:
                        data = {}
                    
                    data['pattern_config'] = {
                        'name': config.name,
                        'tier': tier or config.tier,
                        'emoji': config.emoji,
                        'order_type': config.order_type,
                        'min_volume_ratio': config.min_volume_ratio
                    }
                    
                    return (True, config.side, config.name, data)
            
            except Exception as e:
                logging.error(f"Error testing {config.name}: {e}")
                continue
        
        # Nessun pattern trovato
        return (False, None, None, None)
    
    def enable_pattern(self, key: str) -> bool:
        """Abilita un pattern"""
        if key in self.patterns:
            self.patterns[key].enabled = True
            return True
        return False
    
    def disable_pattern(self, key: str) -> bool:
        """Disabilita un pattern"""
        if key in self.patterns:
            self.patterns[key].enabled = False
            return True
        return False
    
    def get_pattern_info(self, key: str) -> Optional[PatternConfig]:
        """Ottiene info su un pattern"""
        return self.patterns.get(key)
    
    def list_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Lista tutti i pattern con info complete
        Utile per /patterns command
        """
        result = {}
        
        for key, config in self.patterns.items():
            result[key] = {
                'name': config.name,
                'enabled': config.enabled,
                'side': config.side,
                'tier': config.tier,
                'emoji': config.emoji,
                'description': config.description,
                'min_volume_ratio': config.min_volume_ratio,
                'order_type': config.order_type
            }
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche registry"""
        total = len(self.patterns)
        enabled = len([p for p in self.patterns.values() if p.enabled])
        
        by_tier = {
            1: len([p for p in self.patterns.values() if p.tier == 1]),
            2: len([p for p in self.patterns.values() if p.tier == 2]),
            3: len([p for p in self.patterns.values() if p.tier == 3])
        }
        
        by_side = {
            'Buy': len([p for p in self.patterns.values() if p.side == 'Buy']),
            'Sell': len([p for p in self.patterns.values() if p.side == 'Sell']),
            'Both': len([p for p in self.patterns.values() if p.side == 'Both'])
        }
        
        return {
            'total': total,
            'enabled': enabled,
            'disabled': total - enabled,
            'by_tier': by_tier,
            'by_side': by_side
        }


# ============================================
# GLOBAL INSTANCE
# ============================================

# Crea istanza globale (singleton)
PATTERN_REGISTRY = PatternRegistry()


# ============================================
# FUNZIONE DI COMPATIBILITÀ (sostituisce check_patterns)
# ============================================

def check_patterns_unified(df: pd.DataFrame, symbol: str = None) -> Tuple[bool, Optional[str], Optional[str], Optional[Dict]]:
    """
    Versione UNIFICATA di check_patterns().
    Usa il registry invece di if/elif multipli.
    
    VANTAGGI:
    - Codice più pulito (no if/elif giganti)
    - Facile aggiungere nuovi pattern (solo register())
    - Priorità automatica per tier
    - Configurazione centralizzata
    
    Returns:
        (found, side, pattern_name, pattern_data)
    """
    return PATTERN_REGISTRY.detect_all(df, symbol)


# ============================================
# ESEMPIO USO
# ============================================

if __name__ == "__main__":
    # Test del registry
    print(f"Pattern registrati: {len(PATTERN_REGISTRY.patterns)}")
    
    # Pattern abilitati per tier
    tier1 = PATTERN_REGISTRY.get_enabled_patterns(tier=1)
    print(f"Tier 1 patterns: {[p.name for p in tier1.values()]}")
    
    # Statistiche
    stats = PATTERN_REGISTRY.get_stats()
    print(f"\nStatistiche:")
    print(f"  Total: {stats['total']}")
    print(f"  Enabled: {stats['enabled']}")
    print(f"  By Tier: {stats['by_tier']}")
    print(f"  By Side: {stats['by_side']}")
    
    # Disabilita pattern
    PATTERN_REGISTRY.disable_pattern('morning_star')
    
    # Ottieni info
    info = PATTERN_REGISTRY.get_pattern_info('volume_spike_breakout')
    if info:
        print(f"\nVolume Spike config:")
        print(f"  Emoji: {info.emoji}")
        print(f"  Tier: {info.tier}")
        print(f"  Min Volume: {info.min_volume_ratio}x")

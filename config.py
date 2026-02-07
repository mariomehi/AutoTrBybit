import os
import threading

# ----------------------------- CONFIG -----------------------------
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
BYBIT_API_KEY = os.environ.get('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.environ.get('BYBIT_API_SECRET', '')

# Scegli l'ambiente di trading
# 'demo' = Demo Trading (fondi virtuali)
# 'live' = Trading Reale (ATTENZIONE: soldi veri!)
TRADING_MODE = os.environ.get('TRADING_MODE', 'demo')

# ======================== CACHE INSTRUMENT INFO ========================
# Cache globale per evitare chiamate API ripetitive
INSTRUMENT_INFO_CACHE = {}
INSTRUMENT_CACHE_LOCK = threading.Lock()
CACHE_EXPIRY_HOURS = 24  # Le info dei symbol cambiano raramente

# Strategy parameters
#VOLUME_FILTER = True
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
}

TREND_FILTER_ENABLED = True
TREND_FILTER_MODE = 'ema_based'  # 'structure', 'ema_based', 'hybrid', 'pattern_only'

TREND_FILTER_CONFIG = {
    'ema_based': {
        'use_ema60': True,          # Prezzo sopra EMA 60 = uptrend
        'allow_consolidation': True, # OK se sopra EMA 60
        'allow_pullback': True,      # OK se non rompe EMA 60
        'ema60_buffer': 0.98,        # 2% buffer sotto EMA 60
    },
    'hybrid': {
        'use_structure': True,       # Check HH+HL
        'use_ema60': True,           # Check EMA 60
        'require_both': False,       # OR logic (uno dei due)
    },
    'pattern_only': {
        'skip_global': True,         # Ogni pattern decide
    }
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
TRAILING_STOP_ENABLED = True

# ===== ADVANCED TRAILING CONFIG (Multi-Level) =====
TRAILING_CONFIG_ADVANCED = {
    'levels': [
        # Level 1: Attiva presto con buffer largo
        {
            'profit_pct': 0.5,      # Attiva a +0.3% profit
            'ema_buffer': 0.004,    # 0.3% sotto EMA (largo)
            'label': 'Early Protection'
        },
        # Level 2: Standard, buffer medio
        {
            'profit_pct': 0.8,      # +0.5% profit
            'ema_buffer': 0.003,    # 0.2% sotto EMA (medio)
            'label': 'Standard Trail'
        },
        # Level 3: Profit buono, stringi il trailing
        {
            'profit_pct': 1.5,      # +1.0% profit
            'ema_buffer': 0.002,    # 0.1% sotto EMA (stretto)
            'label': 'Tight Trail'
        },
        # Level 4: Grande profit, trailing ultra-stretto
        {
            'profit_pct': 2.5,      # +2.0% profit
            'ema_buffer': 0.001,   # 0.05% sotto EMA (ultra stretto)
            'label': 'Ultra Tight Trail'
        },
    ],
    'never_back': True,         # SL non torna mai indietro
    'check_interval': 30,       # Check ogni 60 secondi (più frequente)
    'min_move_pct': 0.1,        # SL deve muoversi almeno 0.1% per aggiornare
}

# ===== BACKWARD COMPATIBILITY (per codice esistente) =====
TRAILING_CONFIG = {
    'activation_percent': TRAILING_CONFIG_ADVANCED['levels'][0]['profit_pct'],
    'ema_buffer': TRAILING_CONFIG_ADVANCED['levels'][0]['ema_buffer'],
    'never_back': TRAILING_CONFIG_ADVANCED['never_back'],
    'check_interval': TRAILING_CONFIG_ADVANCED['check_interval'],
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

HTF_MOMENTUM_CONFIG = {
    'enabled': True,
    'min_bearish_signals': 2,  # Min segnali bearish per bloccare (2-4)
    'ema10_slope_threshold': -0.1,  # % slope negativo per considerare bearish
    'overextension_threshold': 0.03,  # % sopra EMA 60 = overextended
    'strong_candle_body_min': 0.60,  # Corpo minimo candela forte
}

# Configurazione EMA per diversi timeframe
EMA_CONFIG = {
    # Scalping (5m, 15m) - Focus su EMA veloci
    'scalping': {
        'timeframes': ['5m', '15m'],
        'rules': {
            # MUST 1: Prezzo sopra EMA 10
            'price_above_ema10': False,
            # MUST 2: Prezzo sopra EMA 60 (trend filter) 👈 NUOVO
            'price_above_ema60': True,
            # BONUS: EMA 5 sopra EMA 10 (momentum forte)
            'ema5_above_ema10': True,
            # GOLD: Pattern vicino a EMA 10 (pullback)
            'near_ema10': True  # Opzionale
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
    'update_interval': 7200,  # 2 ore in secondi (12 * 60 * 60)
    'min_volume_usdt': 5000000,  # Min volume 24h: 10M USDT
    'min_price_change': 5.0,  # Min variazione 24h: +5%
    'max_price_change': 110.0,  # Max variazione 24h: +110% (evita pump & dump)
    'exclude_symbols': ['USDCUSDT', 'TUSDUSDT', 'BUSDUSDT'],  # Stablecoins da escludere
    'sorting': 'price_change_percent',  # 'price_change_percent' o 'volume'
}

# Storage per simboli auto-discovered
AUTO_DISCOVERED_SYMBOLS = set()
AUTO_DISCOVERED_LOCK = threading.Lock()

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

# ===== MARKET TIME FILTER =====
MARKET_TIME_FILTER_ENABLED = True
# Ore UTC bloccate (default: 01-04 UTC)
MARKET_TIME_FILTER_BLOCKED_UTC_HOURS = {1, 2}
# Modalità: se True blocca solo autotrade, se False blocca anche analisi pattern
MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY = True

# Aggiungi configurazione pattern-specific
PATTERN_ORDER_TYPE = {
    # Pattern veloci → MARKET (no time to wait)
    'volume_spike_breakout': 'market',
    'liquidity_sweep_reversal': 'market',
    'pin_bar_bullish': 'market',
    
    # Pattern lenti → LIMIT (hai tempo)
    'breakout_retest': 'limit',       # Retest = hai tempo
    'bullish_flag_breakout': 'limit', # Flag = setup lento
    'morning_star': 'limit',          # 3 candele = lento
    'bullish_engulfing': 'limit',     # 2 candele = tempo
}

LIMIT_ORDER_CONFIG = {
    'offset_pct': 0.0015,  # Entry 0.15% SOTTO prezzo corrente
    'timeout_seconds': 60,  # Cancella se non fill in 60s
    'fallback_to_market': True,  # Se timeout → prova market
}

# ===== AGGRESSIVE PROFIT LOCK CONFIG =====
PROFIT_LOCK_ENABLED = True  # Abilita/disabilita profit lock aggressivo
PROFIT_LOCK_CONFIG = {
    'multiplier': 3.0,        # Attiva quando profit >= 3.0x risk iniziale
    'retention': 0.90,        # Trattieni 80% del profit raggiunto
    'min_profit_usd': 20.0,   # Min profit in USD per attivare (evita micro-profit)
}

# ===== MULTI-TP SYSTEM (Dynamic Take Profit) =====
MULTI_TP_ENABLED = True  # Abilita/disabilita sistema multi-TP

MULTI_TP_CONFIG = {
    'enabled': True,
    
    # Livelli TP (in Risk-Reward ratio)
    'levels': [
        {
            'label': 'TP1 - Quick Profit',
            'rr_ratio': 1.0,      # 1.0R dal entry
            'close_pct': 0.50,    # Chiudi 50% posizione
            'emoji': '🎯'
        },
        {
            'label': 'TP2 - Bank It',
            'rr_ratio': 1.5,      # 1.5R dal entry
            'close_pct': 0.30,    # Chiudi 30% posizione
            'emoji': '🎯🎯'
        },
        {
            'label': 'TP3 - Runner',
            'rr_ratio': 2.5,      # 2.5R dal entry
            'close_pct': 0.30,    # Chiudi 20% posizione (residuo)
            'emoji': '🎯🎯🎯'
        }
    ],
    
    # Impostazioni avanzate
    'check_interval': 30,          # Check ogni 30 secondi
    'min_partial_qty': 0.001,      # Min qty per chiusura parziale (BTC)
    'activate_trailing_after_tp1': True,  # Trailing attivo dopo TP1
    'buffer_pct': 0.002,           # Buffer 0.2% per considerar TP "hit"
}

# Tracking TP hit per posizione
TP_TRACKING = {}  # symbol -> {'tp1': False, 'tp2': False, 'tp3': False, 'tp1_qty': 0, ...}
TP_TRACKING_LOCK = threading.Lock()

# ===== RSI + STOCHASTIC RSI CONFIG =====
MOMENTUM_INDICATORS_ENABLED = True  # Master switch

MOMENTUM_THRESHOLDS = {
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'rsi_ideal_min': 40,
    'rsi_ideal_max': 60,
    'stoch_oversold': 20,
    'stoch_overbought': 80
}

# ===== BREAK-EVEN SYSTEM =====
BREAKEVEN_ENABLED = True

BREAKEVEN_CONFIG = {
    # METODO 1: Time-Based (dopo N minuti, sposta SL a break-even)
    'time_based': {
        'enabled': True,
        'minutes': 5,  # Dopo 10 minuti → SL a break-even
        'buffer_pct': 0.001,  # 0.1% sopra entry (per coprire fee)
    },
    
    # METODO 2: Candle-Based (dopo N candele verdi, proteggi)
    'candle_based': {
        'enabled': True,
        'min_green_candles': 2,  # 2 candele verdi consecutive
        'buffer_pct': 0.002,  # 0.2% sopra entry
    },
    
    # METODO 3: Profit-Based (appena in profit minimo)
    'profit_based': {
        'enabled': True,
        'min_profit_pct': 0.3,  # Appena +0.3% profit
        'lock_pct': 0.1,  # Sposta SL a +0.1% (garantisci almeno questo)
    },
    
    # METODO 4: Quick Exit (esci se segnali negativi dopo N min)
    'quick_exit': {
        'enabled': True,
        'check_after_minutes': 5,  # Controlla dopo 5 min
        'exit_if_negative': True,  # Esci se in negativo
        'max_loss_pct': -0.5,  # Max perdita tollerata: -0.5%
    },
    
    # Check interval
    'check_interval': 30,  # Controlla ogni 30 secondi
}

# Pattern Management System
AVAILABLE_PATTERNS = {
    # ═══════════════════════════════════════════
    # TIER 1: HIGH PROBABILITY (60-72% win)
    # ═══════════════════════════════════════════
    'volume_spike_breakout': {
        'name': 'Volume Spike Breakout',
        'enabled': True,  # ✅
        'description': 'Breakout volume 3x+, EMA alignment',
        'side': 'Buy',
        'emoji': '📊💥'
    },
    'breakout_retest': {
        'name': 'Breakout + Retest',
        'enabled': True,  # ✅
        'description': 'Consolidation → Breakout → Retest → Bounce',
        'side': 'Buy',
        'emoji': '🔄📈'
    },
    'triple_touch_breakout': {
        'name': 'Triple Touch Breakout',
        'enabled': True,  # ✅
        'description': '3 tocchi resistance + breakout sopra EMA 60',
        'side': 'Buy',
        'emoji': '🎯3️⃣'
    },
    'liquidity_sweep_reversal': {
        'name': 'Liquidity Sweep + Reversal',
        'enabled': True,  # ✅
        'description': 'Smart money sweep + reversal',
        'side': 'Buy',
        'emoji': '💎'
    },
    
    # ═══════════════════════════════════════════
    # TIER 2: GOOD (52-62% win)
    # ═══════════════════════════════════════════
    'sr_bounce': {
        'name': 'Support/Resistance Bounce',
        'enabled': True,  # ✅
        'description': 'Bounce su S/R con rejection',
        'side': 'Buy',
        'emoji': '🎯'
    },
    'bullish_comeback': {
        'name': 'Bullish Comeback',
        'enabled': True,  # ✅
        'description': 'Inversione dopo tentativo ribassista',
        'side': 'Buy',
        'emoji': '🔄'
    },
    'compression_breakout': {
        'name': 'Compression Breakout (Enhanced)',
        'enabled': True,  # ✅
        'description': 'EMA compression + breakout (RSI, vol, HTF)',
        'side': 'Buy',
        'emoji': '💥'
    },
    'bullish_flag_breakout': {
        'name': 'Bullish Flag Breakout (Enhanced)',
        'enabled': True,  # ✅
        'description': 'Pole + flag + breakout (vol 2x+)',
        'side': 'Buy',
        'emoji': '🚩'
    },
    'morning_star_ema_breakout': {
        'name': 'Morning Star + EMA Breakout',
        'enabled': True,  # ✅
        'description': 'Morning Star + rottura EMA',
        'side': 'Buy',
        'emoji': '⭐💥'
    },
    'higher_low_breakout': {
        'name': 'Higher Low Consolidation Breakout',
        'enabled': True,  # ✅
        'description': 'Impulso + higher lows + breakout',
        'side': 'Buy',
        'emoji': '📈🔺'
    },
        'bud_pattern': {
        'name': 'BUD Pattern',
        'enabled': True,
        'description': 'Breakout + 2 candele riposo nel range',
        'side': 'Buy',
        'emoji': '🌱'
    },
    'maxi_bud_pattern': {
        'name': 'MAXI BUD Pattern',
        'enabled': True,
        'description': 'Breakout + 3+ candele riposo (setup più forte)',
        'side': 'Buy',
        'emoji': '🌟🌱'
    },
    
    # ═══════════════════════════════════════════
    # TIER 3: CLASSIC PATTERNS - USA ENHANCED
    # ═══════════════════════════════════════════
    'bullish_engulfing': {
        'name': 'Bullish Engulfing',
        'enabled': True,  # ✅ MA USA ENHANCED VERSION
        'description': 'Engulfing su EMA (Enhanced)',
        'side': 'Buy',
        'emoji': '🟢'
    },
    'pin_bar_bullish': {
        'name': 'Pin Bar Bullish',
        'enabled': True,  # ✅ MA USA ENHANCED VERSION
        'description': 'Pin bar su EMA (Enhanced)',
        'side': 'Buy',
        'emoji': '📍'
    },
    'morning_star': {
        'name': 'Morning Star',
        'enabled': True,  # ✅ ABILITA + USA ENHANCED VERSION
        'description': '3 candele reversal su EMA (Enhanced)',
        'side': 'Buy',
        'emoji': '⭐'
    },
}

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

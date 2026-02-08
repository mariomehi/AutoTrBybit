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
ATR_MULT_SL = 1.0
ATR_MULT_TP = 1.8
RISK_USD = 10.0

RISK_ADAPTIVE = {
    'enabled': True,
    'base_risk_usd': 10.0,
    
    # Moltiplica risk basato su session trading
    'session_multipliers': {
        'asian': 0.6,     # Ore 0-8 UTC: bassa liquidità
        'london': 1.0,    # Ore 8-13 UTC: liquidità normale
        'overlap': 1.3,   # Ore 13-16 UTC: overlap London+NY (best)
        'newyork': 1.2,   # Ore 16-22 UTC: liquidità alta
        'late_us': 0.8,   # Ore 22-24 UTC: liquidità calante
    },
    
    # Riduci risk durante alta volatilità
    'volatility_adjustment': {
        'enabled': True,
        'atr_threshold_high': 0.005,  # ATR > 0.5% = high volatility
        'atr_threshold_very_high': 0.008,  # ATR > 0.8% = extreme
        'high_vol_multiplier': 0.7,  # Riduci a 70%
        'extreme_vol_multiplier': 0.5,  # Riduci a 50%
    },
    
    # Aumenta risk dopo winning streak
    'streak_adjustment': {
        'enabled': True,
        'min_wins_for_increase': 3,  # 3 win di fila
        'increase_multiplier': 1.2,  # Aumenta a 120%
        'max_multiplier': 1.5,  # Max 150% del base
    },
}

ENABLED_TFS = ['5m']

ATR_MULTIPLIERS_BY_TF = {
    '5m': {'sl': 1.0, 'tp': 2.0, 'minrr': 1.8},  # Era: tp 1.3, minrr 1.2
    '15m': {'sl': 1.2, 'tp': 2.5, 'minrr': 2.0},  
    '1h': {'sl': 1.5, 'tp': 3.5, 'minrr': 2.5},
}

# Flag globale: abilita/disabilita volume filter
VOLUME_FILTER_ENABLED = True  # Default: abilitato
# Modalità volume filter
VOLUME_FILTER_MODE = 'adaptive'  # 'strict', 'adaptive', 'pattern-only'
# Threshold per diversi modi
VOLUME_THRESHOLDS = {
    'strict': 2.5,     # 2.5x per pattern ultra-strong
    'adaptive': 1.5,   # 1.5x per balance qualità/quantità
    'pattern-only': {  # Custom per pattern
        'volume_spike_breakout': 2.0,
        'liquidity_sweep_reversal': 1.8,
        'bud_pattern': 1.5,
    }
}

VOLUME_THRESHOLDS_BY_TF = {
    '5m': 1.5,   # 5m: volume medio-alto (scalping ha volume)
    '15m': 1.8,  # 15m: volume alto
    '1h': 2.0,   # 1h: volume molto alto
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
        # Level 1: MOLTO AGGRESSIVO per 5m
        {
            'profit_pct': 0.3,      # Attiva SUBITO a +0.3%
            'ema_buffer': 0.0015,    # Buffer strettissimo 0.15%
            'label': 'Quick Lock 5m'
        },
        # Level 2: Dopo piccolo profit, stringi subito
        {
            'profit_pct': 0.6,      # +0.6% profit
            'ema_buffer': 0.001,    # 0.1% buffer
            'label': 'Fast Trail 5m'
        },
        # Level 3: Profit decente, ultra-tight
        {
            'profit_pct': 1.0,      # +1.0% profit
            'ema_buffer': 0.0008,    # 0.08% buffer
            'label': 'Tight Trail'
        },
        # Level 4: Grande profit (raro su 5m)
        {
            'profit_pct': 1.5,      # +1.5% profit (max realistico 5m)
            'ema_buffer': 0.0005,   # 0.05% buffer
            'label': 'Ultra Tight Trail'
        },
    ],
    'never_back': True,         # SL non torna mai indietro
    'check_interval': 20,       # Check ogni 20 secondi (più frequente)
    'min_move_pct': 0.05,        # SL deve muoversi almeno 0.1% per aggiornare
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
    '5m': '5m',  # Entry su 5m → EMA 10 da 5m
    '15m': '30m', # Entry su 15m → EMA 10 da 30m
    '30m': '1h',  # Entry su 30m → EMA 10 da 1h
    '1h': '4h',   # Entry su 1h → EMA 10 da 4h
    '4h': '4h',   # Entry su 4h → EMA 10 da 4h stesso
}

# Buffer EMA Stop Loss (% sotto l'EMA per evitare falsi breakout)
EMA_SL_BUFFER = 0.0015  # 0.2% sotto l'EMA
# Esempio: se EMA 10 = $100, SL = $100 * (1 - 0.002) = $99.80
EMA_SL_BUFFER_BY_TF = {
    '5m': 0.0015,   # 0.15% (strettissimo per scalping)
    '15m': 0.0020,  # 0.20%
    '30m': 0.0025,  # 0.25%
    '1h': 0.0030,   # 0.30%
}

# Symbol-specific risk overrides (opzionale)
SYMBOL_RISK_OVERRIDE = {
    # Crypto volatili = risk ridotto
    'DOGEUSDT': 5.0,   # Doge molto volatile
    'SHIBUSDT': 5.0,   # Shib pump/dump
    'PEPEUSDT': 5.0,   # Meme coin volatili
    
    # Crypto stabili = risk normale/aumentato
    'BTCUSDT': 12.0,   # BTC più stabile, risk 20% in più
    'ETHUSDT': 11.0,   # ETH stabile
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
    'top_count': 7,  # Top 10 symbols
    'timeframe': '5m',  # Timeframe da analizzare
    'autotrade': True,  # Autotrade per auto-discovery (False = solo notifiche)
    'update_interval': 3600,  # 1 ora in secondi (12 * 60 * 60)
    'min_volume_usdt': 5000000,  # Min volume 24h: 10M USDT
    'min_price_change': 3.0,  # Min variazione 24h: +3%
    'max_price_change': 120.0,  # Max variazione 24h: +110% (evita pump & dump)
    'min_trades_24h': 50000,  # Min 50k trades 24h (liquidità profondità)
    'max_spread_pct': 0.05,  # Max 0.05% spread (evita low liquidity)
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
MARKET_TIME_FILTER_BLOCKED_UTC_HOURS = {0, 1, 2, 3, 4}
# Modalità: se True blocca solo autotrade, se False blocca anche analisi pattern
MARKET_TIME_FILTER_BLOCK_AUTOTRADE_ONLY = False  
TRADING_HOURS_CONFIG = {
    'timezone': 'UTC',
    'allowed_hours': list(range(8, 22)),  # 08:00-21:59 UTC (London + NY)
    'best_hours': list(range(13, 16)),    # 13:00-15:59 UTC (Overlap)
    'best_hours_multiplier': 1.3,  # Risk 30% in più durante overlap
}

# Aggiungi configurazione pattern-specific
PATTERN_ORDER_TYPE = {
    'volume_spike_breakout': 'market',
    'liquidity_sweep_reversal': 'market',
    'pin_bar_bullish': 'market',
    'bud_pattern': 'market',  # Veloce anche questo
    'sr_bounce': 'market',
    'bullish_comeback': 'market',
    'bullish_engulfing': 'limit',
}

LIMIT_ORDER_CONFIG = {
    'offset_pct': 0.0015,  # Entry 0.15% SOTTO prezzo corrente
    'timeout_seconds': 60,  # Cancella se non fill in 60s
    'fallback_to_market': True,  # Se timeout → prova market
}

# ===== AGGRESSIVE PROFIT LOCK CONFIG =====
PROFIT_LOCK_ENABLED = True  # Abilita/disabilita profit lock aggressivo
PROFIT_LOCK_CONFIG = {
    'multiplier': 2.0,        # Attiva quando profit >= 3.0x risk iniziale
    'retention': 0.80,        # Trattieni 80% del profit raggiunto
    'min_profit_usd': 10.0,   # Min profit in USD per attivare (evita micro-profit)
        # AGGIUNGI: Livelli multipli
    'levels': [
        {'multiplier': 2.0, 'retention': 0.70, 'label': 'Quick Lock'},
        {'multiplier': 3.0, 'retention': 0.80, 'label': 'Strong Lock'},
        {'multiplier': 4.0, 'retention': 0.90, 'label': 'Full Lock'},
    ]
}

# ===== MULTI-TP SYSTEM (Dynamic Take Profit) =====
MULTI_TP_ENABLED = True  # Abilita/disabilita sistema multi-TP

MULTI_TP_CONFIG = {
    'enabled': True,
    
    # Livelli TP (in Risk-Reward ratio)
    'levels': [
        {
            'label': 'TP1 - Quick Scalp',
            'rr_ratio': 0.8,      # Più veloce, target +0.8R
            'close_pct': 0.50,    # Chiudi 50% posizione
            'emoji': '🎯'
        },
        {
            'label': 'TP2 - Bank It',
            'rr_ratio': 1.2,      # Realistico per 5m
            'close_pct': 0.30,    # Chiudi 30% posizione
            'emoji': '🎯🎯'
        },
        {
            'label': 'TP3 - Runner',
            'rr_ratio': 1.8,      # Max realistico 5m
            'close_pct': 0.20,    # Chiudi 20% posizione (residuo)
            'emoji': '🎯🎯🎯'
        }
    ],
    
    # Impostazioni avanzate
    'check_interval': 15,          # Check ogni 15 secondi
    'min_partial_qty': 0.001,      # Min qty per chiusura parziale (BTC)
    'activate_trailing_after_tp1': True,  # Trailing attivo dopo TP1
    'buffer_pct': 0.003,           # Buffer 0.2% per considerar TP "hit"
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
        'minutes': 3,  # Dopo 3 minuti → SL a break-even
        'buffer_pct': 0.0005,  # 0.05% sopra entry (per coprire fee)
    },
    
    # METODO 2: Candle-Based (dopo N candele verdi, proteggi)
    'candle_based': {
        'enabled': True,
        'min_green_candles': 3,  # 2 candele verdi consecutive
        'buffer_pct': 0.0008,  # 0.2% sopra entry
    },
    
    # METODO 3: Profit-Based (appena in profit minimo)
    'profit_based': {
        'enabled': True,
        'min_profit_pct': 0.25,  # Appena +0.25% profit (velocissimo)
        'lock_pct': 0.08,  # Blocca +0.08% (copre fee + small profit)
    },
    
    # METODO 4: Quick Exit (esci se segnali negativi dopo N min)
    'quick_exit': {
        'enabled': True,
        'check_after_minutes': 3,  # Controlla dopo 5 min
        'exit_if_negative': True,  # Esci se in negativo
        'max_loss_pct': -0.4,  # Max -0.4% tollerato
    },
    
    # Check interval
    'check_interval': 20,  # Controlla ogni 30 secondi
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
        'min_timeframes': ['5m', '15m'],  # AGGIUNGI questo
        'priority': 1,  # AGGIUNGI priority per ordinamento
        'side': 'Buy',
        'emoji': '📊💥'
    },
    'liquidity_sweep_reversal': {
        'name': 'Liquidity Sweep + Reversal',
        'enabled': True,  # ✅
        'description': 'Smart money sweep + reversal',
        'min_timeframes': ['5m', '15m'],
        'priority': 2,
        'side': 'Buy',
        'emoji': '💎'
    },
    'pin_bar_bullish': {
        'name': 'Pin Bar Bullish',
        'enabled': True,  # ✅ FAST (single candle)
        'description': 'Pin bar su EMA (Enhanced)',
        'min_timeframes': ['5m'],
        'priority': 3,
        'side': 'Buy',
        'emoji': '📍'
    },
    'bud_pattern': {
        'name': 'BUD Pattern',
        'enabled': True, # ✅ OK per 5m
        'description': 'Breakout + 2 candele riposo nel range',
        'min_timeframes': ['5m', '15m'],
        'priority': 4,
        'side': 'Buy',
        'emoji': '🌱'
    },
    'breakout_retest': {
        'name': 'Breakout + Retest',
        'enabled': False,  # ❌ Troppo lento per 5m (richiede consolidamento)
        'description': 'Consolidation → Breakout → Retest → Bounce',
        'min_timeframes': ['15m', '30m'],  # Solo timeframes più alti
        'side': 'Buy',
        'emoji': '🔄📈'
    },
    'triple_touch_breakout': {
        'name': 'Triple Touch Breakout',
        'enabled': False,  # ❌ Richiede 3+ touch = tempo
        'description': '3 tocchi resistance + breakout sopra EMA 60',
        'min_timeframes': ['15m'],
        'side': 'Buy',
        'emoji': '🎯3️⃣'
    },
    
    # ═══════════════════════════════════════════
    # TIER 2: GOOD (52-62% win)
    # ═══════════════════════════════════════════
    'sr_bounce': {
        'name': 'Support/Resistance Bounce',
        'enabled': True,  # ✅ Bounce veloce
        'description': 'Bounce su S/R con rejection',
        'min_timeframes': ['5m'],
        'priority': 5,
        'side': 'Buy',
        'emoji': '🎯'
    },
    'bullish_comeback': {
        'name': 'Bullish Comeback',
        'enabled': True,  # ✅ Reversal veloce
        'description': 'Inversione dopo tentativo ribassista',
        'min_timeframes': ['5m'],
        'priority': 6,
        'side': 'Buy',
        'emoji': '🔄'
    },
    'bullish_engulfing': {
        'name': 'Bullish Engulfing',
        'enabled': True,  # ✅ 2 candele OK
        'description': 'Engulfing su EMA (Enhanced)',
        'min_timeframes': ['5m'],
        'priority': 7,
        'side': 'Buy',
        'emoji': '🟢'
    },
    'compression_breakout': {
        'name': 'Compression Breakout (Enhanced)',
        'enabled': False,  # ❌ Richiede compressione lunga
        'description': 'EMA compression + breakout (RSI, vol, HTF)',
        'min_timeframes': ['15m', '30m'],
        'side': 'Buy',
        'emoji': '💥'
    },
    'bullish_flag_breakout': {
        'name': 'Bullish Flag Breakout (Enhanced)',
        'enabled': False,  # ❌ Consolidamento lento
        'description': 'Pole + flag + breakout (vol 2x+)',
        'min_timeframes': ['30m', '1h'],
        'side': 'Buy',
        'emoji': '🚩'
    },
    'morning_star_ema_breakout': {
        'name': 'Morning Star + EMA Breakout',
        'enabled': False,  # ❌ Combinazione lenta
        'description': 'Morning Star + rottura EMA',
        'min_timeframes': ['15m'],
        'side': 'Buy',
        'emoji': '⭐💥'
    },
    'higher_low_breakout': {
        'name': 'Higher Low Consolidation Breakout',
        'enabled': False,  # ❌ Richiede consolidamento
        'description': 'Impulso + higher lows + breakout',
        'min_timeframes': ['30m'],
        'side': 'Buy',
        'emoji': '📈🔺'
    },
    'maxi_bud_pattern': {
        'name': 'MAXI BUD Pattern',
        'enabled': False, # ❌ Richiede 3+ candele riposo (15+ min)
        'description': 'Breakout + 3+ candele riposo (setup più forte)',
        'min_timeframes': ['15m'],
        'side': 'Buy',
        'emoji': '🌟🌱'
    },
    
    # ═══════════════════════════════════════════
    # TIER 3: CLASSIC PATTERNS - USA ENHANCED
    # ═══════════════════════════════════════════
    'morning_star': {
        'name': 'Morning Star',
        'enabled': False,  # ❌ 3 candele = 15 minuti (troppo)
        'description': '3 candele reversal su EMA (Enhanced)',
        'min_timeframes': ['15m'],
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

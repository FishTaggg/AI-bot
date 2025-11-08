# crypto_ai_bot_auto.py
# Single-file bot: data collection, indicators, auto-train XGBoost, realtime inference, Discord alerts + commands
# NOTE: configure environment variables before running (see README comments below)

import os
import asyncio
import aiohttp
import discord
import json
import logging
import math
from discord.ext import tasks, commands
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

# ---------------- CONFIG ----------------
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
BINANCE_BASE = "https://api.binance.com"
VS_CURRENCY = "usd"

DATA_FILE = "crypto_dataset.csv"         # aggregated dataset for training
MODEL_FILE = "crypto_xgb_model.joblib"   # persisted model
WATCHLIST_FILE = "watchlist.json"
LOG_FILE = "bot.log"

MAX_PAGES = 5            # number of CoinGecko pages to scan (250 per page) -> adjust for coverage
MARKETCAP_MIN = 5_000    # lower marketcap threshold (tunable)
MARKETCAP_MAX = 200_000_000
POLL_INTERVAL_MIN = 5    # minutes between automatic scans
TRAIN_EVERY_HOURS = 24   # automatic retrain cadence
NEXT_HORIZON_HOURS = 4   # label horizon (predict whether price rises in next 4 hours)
LABEL_PCT_THRESHOLD = 0.03  # label positive if next_horizon_pct >= 3%

ALERT_SCORE = 85
WATCH_SCORE = 60
ALERT_COOLDOWN_HOURS = 6

# Environment variables (set these)
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
DM_USER_ID = os.getenv("DISCORD_DM_USER_ID")  # optional
BINANCE_API_KEY = os.getenv("P2iRn4qZd2Iec4MB6dpLAtnZsnSZZdrJ9Y4XAw4GOO6ysmfuR54LDBMhyhHhMD4C")      # optional
BINANCE_API_SECRET = os.getenv("M5g4Sp2a9rbGQDBKNnlOYY0HQt6Qw68A8n5KMv686lufKzJ6s42cix0Y2qAalKbI")# optional

if not BOT_TOKEN or CHANNEL_ID == 0:
    raise SystemExit("ERROR: set DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID environment variables before running")

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename=LOG_FILE, filemode="a")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# ---------------- discord bot ----------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

STABLE_KEYWORDS = ['usd', 'usdt', 'usdc', 'busd', 'tusd', 'dai', 'usdn', 'gusd', 'usdp']

# ---------------- utilities ----------------
def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(watchlist, f, indent=2, ensure_ascii=False)

watchlist = load_watchlist()

def iso_now():
    return datetime.utcnow().isoformat()

def hours_since_iso(iso_ts):
    try:
        dt = datetime.fromisoformat(iso_ts)
        return (datetime.utcnow() - dt).total_seconds() / 3600.0
    except Exception:
        return float("inf")

def should_alert(symbol):
    entry = watchlist.get(symbol.upper())
    if not entry:
        return True
    last_ts = entry.get("timestamp")
    if not last_ts:
        return True
    return hours_since_iso(last_ts) > ALERT_COOLDOWN_HOURS

# ---------------- indicators & scoring (your existing logic) ----------------
def compute_technical_indicators(df):
    if df is None or len(df) < 30:
        return {}
    close = df['close'].ffill()
    vol = df['volume'].fillna(0)
    ema7 = close.ewm(span=7, adjust=False).mean()
    ema25 = close.ewm(span=25, adjust=False).mean()
    try:
        ema_cross = (ema7.iloc[-1] > ema25.iloc[-1]) and (ema7.iloc[-2] <= ema25.iloc[-2])
    except Exception:
        ema_cross = False
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1]) if not rsi.empty else None
    if len(vol) >= 8:
        mean_prev7 = vol.iloc[-8:-1].mean()
        vol_spike = float(vol.iloc[-1] / (mean_prev7 + 1e-9))
    else:
        vol_spike = float(vol.iloc[-1] / (vol.mean() + 1e-9))
    lookback = 24
    try:
        prev_max = df['high'].iloc[-(lookback+1):-1].max()
        last_close = close.iloc[-1]
        breakout = (last_close > prev_max) and (close.iloc[-2] < prev_max)
    except Exception:
        breakout = False
    return {'ema_cross': ema_cross, 'vol_spike': vol_spike, 'rsi': rsi_val, 'breakout': breakout}

def compute_orderbook_imbalance(orderbook, top_n=20):
    try:
        bids = orderbook.get('bids', [])[:top_n]
        asks = orderbook.get('asks', [])[:top_n]
        bids_sum = sum(float(p) * float(q) for p,q in bids)
        asks_sum = sum(float(p) * float(q) for p,q in asks)
        imbalance = bids_sum / (asks_sum + 1e-9)
        total = bids_sum + asks_sum
        return imbalance, total
    except Exception:
        return None, 0

def compute_score(base_score, indicators, price_change_24h, market_cap, social_score=0):
    score = base_score
    if indicators.get('ema_cross'):
        score += 18
    if indicators.get('breakout'):
        score += 20
    if indicators.get('vol_spike') and indicators['vol_spike'] > 1.8:
        score += min(14, (indicators['vol_spike'] - 1.0) * 6)
    rsi = indicators.get('rsi')
    if rsi is not None:
        if 30 <= rsi <= 55:
            score += 6
        elif rsi < 30:
            score += 3
    obi = indicators.get('orderbook_imbalance')
    if obi is not None:
        if obi > 1.5:
            score += min(12, (obi - 1.0) * 6)
    if market_cap > 0:
        mc_factor = max(0, (1e7 / market_cap)) * 4
        score += min(mc_factor, 12)
    score += social_score * 0.08
    score += max(min(price_change_24h, 50), -10) * 0.12
    score = max(0, min(100, score))
    reasons = []
    if indicators.get('ema_cross'): reasons.append("EMA7>EMA25 (recent cross)")
    if indicators.get('breakout'): reasons.append("Breakout above recent resistance")
    if indicators.get('vol_spike') and indicators['vol_spike']>1:
        reasons.append(f"Vol spike x{indicators['vol_spike']:.1f}")
    if rsi is not None: reasons.append(f"RSI {rsi:.0f}")
    if obi is not None and obi>1.2: reasons.append(f"Orderbook buy imbalance x{obi:.2f}")
    if price_change_24h is not None: reasons.append(f"24h {price_change_24h:+.2f}%")
    return {'score': score, 'reasons': reasons}

def compute_risk(price_change_24h, volume, market_cap, orderbook_liquidity=0):
    try:
        liquidity_ratio = market_cap / max(volume, 1)
    except Exception:
        liquidity_ratio = float("inf")
    risk_flag = False
    if price_change_24h and price_change_24h > 200 and liquidity_ratio < 50:
        risk_flag = True
    if market_cap < 50_000:
        risk_flag = True
    if orderbook_liquidity and orderbook_liquidity < 5000:
        risk_flag = True
    return risk_flag

def compute_trade_levels(current_price, score, indicators):
    if current_price is None or current_price == 0:
        return {}
    breakout = indicators.get('breakout', False)
    vol_spike = indicators.get('vol_spike', 1.0)
    if breakout:
        entry = current_price * 0.995
        entry_note = "Breakout detected — suggested entry on small retest (~-0.5%)"
    else:
        entry = current_price * 1.0015
        entry_note = "Momentum entry — buy slightly above current to confirm move"
    s = max(0, min(100, score))
    base_tp1 = 0.03; base_tp2 = 0.07; base_tp3 = 0.15
    tp_boost = (s / 100.0) * 0.08
    tp1_pct = base_tp1 + tp_boost * 0.25
    tp2_pct = base_tp2 + tp_boost * 0.6
    tp3_pct = base_tp3 + tp_boost * 1.0
    tp1 = entry * (1 + tp1_pct); tp2 = entry * (1 + tp2_pct); tp3 = entry * (1 + tp3_pct)
    sl_pct = 0.08 - (s / 100.0) * 0.06
    sl_pct = max(0.02, min(0.08, sl_pct))
    sl = entry * (1 - sl_pct)
    trail_base = 0.03
    trail_tightness = (1.0 - (s / 150.0))
    tl_pct = max(0.015, trail_base * trail_tightness + (max(0, vol_spike - 1.0) * 0.005))
    tl_activate_at = tp2
    notes = (
        f"{entry_note}. SL set at -{sl_pct*100:.1f}%. "
        f"Partial at TP1 ({tp1_pct*100:.1f}%), more at TP2 ({tp2_pct*100:.1f}%), final at TP3 ({tp3_pct*100:.1f}%). "
        f"Activate trailing at TP2; trailing: {tl_pct*100:.2f}%."
    )
    return {
        'entry': float(entry), 'tp1': float(tp1), 'tp2': float(tp2), 'tp3': float(tp3),
        'sl': float(sl), 'sl_pct': float(sl_pct), 'tl_pct': float(tl_pct),
        'tl_activate_at': float(tl_activate_at), 'notes': notes
    }

# ---------------- HTTP helpers ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                logging.debug(f"fetch_json non-200 {resp.status} for {url}")
                return None
    except Exception as e:
        logging.debug(f"fetch_json error {e} for {url}")
        return None

# ---------------- Binance / CoinGecko fetchers ----------------
async def fetch_binance_klines(session, symbol_pair, limit=200, interval="1h"):
    url = f"{BINANCE_BASE}/api/v3/klines?symbol={symbol_pair}&interval={interval}&limit={limit}"
    data = await fetch_json(session, url)
    if not data:
        return None
    cols = ['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base','taker_quote','ignore']
    df = pd.DataFrame(data, columns=cols)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[['open_time','open','high','low','close','volume']]

async def fetch_binance_order_book(session, symbol_pair, limit=100):
    url = f"{BINANCE_BASE}/api/v3/depth?symbol={symbol_pair}&limit={limit}"
    return await fetch_json(session, url)

async def fetch_coingecko_social(session, coin_id):
    url = f"{COINGECKO_BASE}/coins/{coin_id}?localization=false&tickers=false&market_data=false&community_data=true&developer_data=false&sparkline=false"
    data = await fetch_json(session, url)
    if not data:
        return 0, {}
    cd = data.get('community_data', {}) or {}
    sentiment_up = data.get('sentiment_votes_up_percentage') or 0
    twitter_followers = cd.get('twitter_followers') or 0
    reddit_subs = cd.get('reddit_subscribers') or 0
    tg = cd.get('telegram_channel_user_count') or 0
    social_score = (twitter_followers / 1000.0) + (reddit_subs / 100.0) + (tg / 500.0) + (sentiment_up / 2.0)
    return social_score, {'twitter_followers': twitter_followers, 'reddit_subs': reddit_subs, 'telegram': tg, 'sentiment_up_pct': sentiment_up}

async def fetch_coins_page(session, page):
    url = f"{COINGECKO_BASE}/coins/markets?vs_currency={VS_CURRENCY}&order=market_cap_desc&per_page=250&page={page}&price_change_percentage=24h"
    data = await fetch_json(session, url)
    return data or []

# ---------------- dataset helpers & labeling ----------------
def append_to_dataset(rows):
    """rows: list of dicts -> append to CSV dataset"""
    df = pd.DataFrame(rows)
    if df.empty:
        return
    if os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(DATA_FILE, index=False)

def build_features_from_row(row):
    """Given coin info & indicators, produce numeric feature vector for model"""
    f = {
        'score': row.get('score', 0),
        'vol_spike': row.get('indicators', {}).get('vol_spike', 0),
        'rsi': row.get('indicators', {}).get('rsi', 50),
        'ema_cross': 1 if row.get('indicators', {}).get('ema_cross') else 0,
        'breakout': 1 if row.get('indicators', {}).get('breakout') else 0,
        'orderbook_imbalance': row.get('indicators', {}).get('orderbook_imbalance') or 0,
        'market_cap': row.get('market_cap') or 0,
        'price': row.get('current_price') or 0,
        'price_change_24h': row.get('price_change_24h') or 0,
        'social_score': row.get('social_score') or 0
    }
    return f

# ---------------- model training & inference ----------------
def train_model_from_csv():
    """Load DATA_FILE, build features & labels, train XGBoost classifier, save model."""
    if not os.path.exists(DATA_FILE):
        logging.info("No data file to train from.")
        return None, None
    df = pd.read_csv(DATA_FILE)
    # expected columns: price, next_horizon_pct (label), and features saved earlier
    # if dataset was appended as raw, ensure necessary columns exist
    required = ['price','next_horizon_pct']
    for c in required:
        if c not in df.columns:
            logging.info("Dataset missing required columns for training.")
            return None, None
    # Label: positive if next_horizon_pct >= LABEL_PCT_THRESHOLD
    df['label'] = (df['next_horizon_pct'] >= LABEL_PCT_THRESHOLD).astype(int)
    # Build features (try to pick existing columns or derive)
    feature_cols = []
    # prefer explicit feature columns if present, else use fallback
    candidates = ['score','vol_spike','rsi','ema_cross','breakout','orderbook_imbalance','market_cap','price_change_24h','social_score']
    for c in candidates:
        if c in df.columns:
            feature_cols.append(c)
    if not feature_cols:
        logging.info("No feature columns present for training.")
        return None, None
    X = df[feature_cols].fillna(0)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    # evaluate
    yproba = clf.predict_proba(X_test)[:,1]
    ypred = clf.predict(X_test)
    try:
        auc = roc_auc_score(y_test, yproba)
    except Exception:
        auc = None
    acc = accuracy_score(y_test, ypred)
    joblib.dump({'model': clf, 'features': feature_cols}, MODEL_FILE)
    logging.info(f"Trained model saved. AUC={auc}, ACC={acc}")
    return clf, feature_cols

def load_model():
    if not os.path.exists(MODEL_FILE):
        return None, None
    data = joblib.load(MODEL_FILE)
    return data.get('model'), data.get('features')

def model_predict_proba(model, feature_cols, row):
    if model is None or feature_cols is None:
        return None
    fdict = build_features_from_row(row)
    X = [fdict.get(c, 0) for c in feature_cols]
    try:
        proba = model.predict_proba([X])[0][1]
        return float(proba)
    except Exception as e:
        logging.debug(f"model predict error: {e}")
        return None

# ---------------- analysis pipeline (main coins scan) ----------------
async def analyze_once(session, model=None, feature_cols=None):
    coins_out = []
    pages = await asyncio.gather(*[fetch_coins_page(session, p) for p in range(1, MAX_PAGES+1)])
    for data in pages:
        if not data:
            continue
        for c in data:
            try:
                cid = c.get('id','').lower()
                sym = c.get('symbol','').lower()
                name = c.get('name','')
                market_cap = c.get('market_cap') or 0
                price_change_24h = c.get('price_change_percentage_24h') or 0
                if any(k in cid for k in STABLE_KEYWORDS) or any(k in sym for k in STABLE_KEYWORDS):
                    continue
                if market_cap < MARKETCAP_MIN or market_cap > MARKETCAP_MAX:
                    continue
                if abs(price_change_24h) < 0.8:
                    # keep threshold lower to gather more data for AI
                    pass
                base_symbol = c.get('symbol','').upper()
                possible_pairs = [f"{base_symbol}USDT", f"{base_symbol}BUSD", f"{base_symbol}USDC", f"{base_symbol}BTC"]
                klines_df = None
                pair_used = None
                for p in possible_pairs:
                    klines_df = await fetch_binance_klines(session, p, limit=200, interval="1h")
                    if klines_df is not None:
                        pair_used = p
                        break
                    await asyncio.sleep(0.12)
                if klines_df is None:
                    continue
                orderbook = None
                social_score = 0
                social_raw = {}
                try:
                    ob_task = fetch_binance_order_book(session, pair_used, limit=100)
                    sc_task = fetch_coingecko_social(session, cid)
                    orderbook, (social_score, social_raw) = await asyncio.gather(ob_task, sc_task)
                except Exception:
                    pass
                indicators = compute_technical_indicators(klines_df)
                if orderbook:
                    obi, ob_liq = compute_orderbook_imbalance(orderbook, top_n=20)
                    if obi is not None:
                        indicators['orderbook_imbalance'] = obi
                else:
                    obi = None; ob_liq = 0
                # base score heuristics similar to earlier
                try:
                    recent = klines_df['close'].iloc[-4:].pct_change().dropna()
                    momentum = float(recent.sum() * 100)
                    base_score = max(0, min(40, momentum * 2 + max(min(price_change_24h, 40), 0) * 0.3))
                except Exception:
                    base_score = max(0, min(40, price_change_24h * 0.6))
                score_info = compute_score(base_score, indicators, price_change_24h, market_cap, social_score=social_score)
                risk_flag = compute_risk(price_change_24h, c.get('total_volume', 0), market_cap, orderbook_liquidity=ob_liq)
                vol_spike = indicators.get('vol_spike') or 0
                if vol_spike > 6 and (obi is not None and obi > 4) and social_score < 2:
                    risk_flag = True
                trade_levels = compute_trade_levels(c.get('current_price'), score_info['score'], indicators)
                # AI prediction
                ai_conf = None
                if model is not None and feature_cols is not None:
                    ai_conf = model_predict_proba(model, feature_cols, {
                        'score': score_info['score'],
                        'indicators': indicators,
                        'market_cap': market_cap,
                        'current_price': c.get('current_price'),
                        'price_change_24h': price_change_24h,
                        'social_score': social_score
                    })
                coin = {
                    'name': name,
                    'symbol': base_symbol,
                    'pair': pair_used,
                    'current_price': c.get('current_price'),
                    'price_change_24h': price_change_24h,
                    'score': score_info['score'],
                    'reasons': score_info['reasons'],
                    'coingecko_url': f"https://www.coingecko.com/en/coins/{c.get('id')}",
                    'exchanges': [pair_used.split(base_symbol)[-1] + " (Binance)" ] if pair_used else [],
                    'risk_flag': risk_flag,
                    'indicators': indicators,
                    'social_raw': social_raw,
                    'social_score': social_score,
                    'trade_levels': trade_levels,
                    'market_cap': market_cap,
                    'ai_confidence': ai_conf
                }
                if coin['score'] >= WATCH_SCORE and should_alert(coin['symbol']):
                    coins_out.append(coin)
            except Exception as e:
                logging.debug(f"error processing coin {c.get('id')}: {e}")
                continue
    # sort by combined metric (score + ai_confidence*30 for example)
    def combined_rank(x):
        ai = x.get('ai_confidence') or 0
        return x.get('score',0) + ai * 30
    return sorted(coins_out, key=combined_rank, reverse=True)

# ---------------- discord embed & sending ----------------
def create_embed_for_coin(coin):
    embed = discord.Embed(
        title=f"{coin['name']} ({coin['symbol']})",
        url=coin.get('coingecko_url'),
        timestamp=datetime.utcnow()
    )
    embed.add_field(name="Pair", value=coin.get('pair','N/A'), inline=True)
    p = coin.get('current_price')
    embed.add_field(name="Price", value=f"${p:,.6f}" if isinstance(p, (int,float)) and p<1 else (f"${p:,.4f}" if p else "N/A"), inline=True)
    embed.add_field(name="24h %", value=f"{coin.get('price_change_24h'):+.2f}%", inline=True)
    embed.add_field(name="Score", value=f"{coin.get('score'):.1f}", inline=True)
    reasons = ", ".join(coin.get('reasons', [])) or "N/A"
    embed.add_field(name="Reasons", value=reasons, inline=False)
    ind = coin.get('indicators', {})
    ind_lines = []
    if 'ema_cross' in ind: ind_lines.append(f"EMA cross: {'YES' if ind['ema_cross'] else 'no'}")
    if 'vol_spike' in ind: ind_lines.append(f"Vol spike: x{ind['vol_spike']:.2f}")
    if 'rsi' in ind and ind['rsi'] is not None: ind_lines.append(f"RSI: {ind['rsi']:.0f}")
    if 'breakout' in ind: ind_lines.append(f"Breakout: {'YES' if ind['breakout'] else 'no'}")
    if 'orderbook_imbalance' in ind: ind_lines.append(f"OBI: x{ind['orderbook_imbalance']:.2f}")
    if ind_lines:
        embed.add_field(name="Tech", value="\n".join(ind_lines), inline=False)
    social = coin.get('social_raw', {})
    s_lines = []
    if social.get('twitter_followers'): s_lines.append(f"Twitter: {social['twitter_followers']:,}")
    if social.get('reddit_subs'): s_lines.append(f"Reddit: {social['reddit_subs']:,}")
    if social.get('telegram'): s_lines.append(f"Telegram: {social['telegram']:,}")
    if s_lines:
        embed.add_field(name="Social", value=" | ".join(s_lines), inline=False)
    tl = coin.get('trade_levels') or {}
    if tl:
        levels_text = []
        if tl.get('entry'):
            levels_text.append(f"Entry: ${tl['entry']:.6f}" if tl['entry'] < 1 else f"Entry: ${tl['entry']:.4f}")
        if tl.get('tp1'): levels_text.append(f"TP1: ${tl['tp1']:.6f}" if tl['tp1'] < 1 else f"TP1: ${tl['tp1']:.4f}")
        if tl.get('tp2'): levels_text.append(f"TP2: ${tl['tp2']:.6f}" if tl['tp2'] < 1 else f"TP2: ${tl['tp2']:.4f}")
        if tl.get('tp3'): levels_text.append(f"TP3: ${tl['tp3']:.6f}" if tl['tp3'] < 1 else f"TP3: ${tl['tp3']:.4f}")
        if tl.get('sl'): levels_text.append(f"SL: ${tl['sl']:.6f}" if tl['sl'] < 1 else f"SL: ${tl['sl']:.4f}")
        if tl.get('tl_pct') is not None:
            levels_text.append(f"Trailing: {tl['tl_pct']*100:.2f}% (activate @ ${tl['tl_activate_at']:.4f})")
        embed.add_field(name="Trade Levels", value="\n".join(levels_text), inline=False)
    ai_conf = coin.get('ai_confidence')
    if ai_conf is not None:
        embed.add_field(name="AI Confidence", value=f"{ai_conf*100:.1f}%", inline=True)
    if coin.get('risk_flag'):
        embed.colour = discord.Colour.dark_red()
        embed.set_footer(text="⚠️ High Risk / Liquidity Concerns")
    elif coin.get('score') >= ALERT_SCORE:
        embed.colour = discord.Colour.red()
        embed.set_footer(text="🚀 High Alert")
    elif coin.get('score') >= WATCH_SCORE:
        embed.colour = discord.Colour.gold()
        embed.set_footer(text="🔎 Watchlist")
    else:
        embed.colour = discord.Colour.blue()
        embed.set_footer(text="ℹ️ Info")
    return embed

async def send_coin_alert(coin):
    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        logging.warning("Discord channel not found")
        return
    embed = create_embed_for_coin(coin)
    await channel.send(embed=embed)
    if DM_USER_ID and (coin['score'] >= ALERT_SCORE or coin.get('risk_flag')):
        try:
            user = await bot.fetch_user(int(DM_USER_ID))
            if user:
                await user.send(embed=embed)
        except Exception as e:
            logging.debug(f"failed to DM user: {e}")

# ---------------- background polling & training tasks ----------------
@tasks.loop(minutes=POLL_INTERVAL_MIN)
async def poll_market():
    await bot.wait_until_ready()
    logging.info("Polling market for signals...")
    model, feature_cols = load_model()
    async with aiohttp.ClientSession() as session:
        coins = await analyze_once(session, model=model, feature_cols=feature_cols)
    if not coins:
        logging.info("No candidates found this round.")
        return
    for coin in coins[:10]:
        try:
            await send_coin_alert(coin)
            watchlist[coin['symbol']] = {
                'score': coin['score'],
                'last_price': coin.get('current_price'),
                'risk_flag': coin.get('risk_flag', False),
                'timestamp': iso_now(),
                'trade_levels': coin.get('trade_levels', {}),
                'ai_confidence': coin.get('ai_confidence')
            }
        except Exception as e:
            logging.debug(f"failed to send alert for {coin['symbol']}: {e}")
    save_watchlist(watchlist)

@tasks.loop(hours=TRAIN_EVERY_HOURS)
async def auto_train_task():
    await bot.wait_until_ready()
    logging.info("Auto-train task triggered.")
    # The training function is synchronous (CPU-bound). Run in executor
    loop = asyncio.get_event_loop()
    def train_sync():
        model, features = train_model_from_csv()
        return model is not None
    ok = await loop.run_in_executor(None, train_sync)
    if ok:
        logging.info("Auto-training complete.")
    else:
        logging.info("Auto-training skipped or failed.")

# ---------------- command: live check single symbol ----------------
async def live_check_symbol(symbol):
    symbol_uc = symbol.upper()
    async with aiohttp.ClientSession() as session:
        pages = await asyncio.gather(*[fetch_coins_page(session, p) for p in range(1, MAX_PAGES+1)])
        target = None
        for data in pages:
            if not data:
                continue
            for c in data:
                if c.get('symbol','').upper() == symbol_uc:
                    target = c
                    break
            if target:
                break
        if not target:
            return None
        c = target
        cid = c.get('id','').lower()
        base_symbol = c.get('symbol','').upper()
        possible_pairs = [f"{base_symbol}USDT", f"{base_symbol}BUSD", f"{base_symbol}USDC", f"{base_symbol}BTC"]
        klines_df = None; pair_used = None
        for p in possible_pairs:
            klines_df = await fetch_binance_klines(session, p, limit=200, interval="1h")
            if klines_df is not None:
                pair_used = p
                break
            await asyncio.sleep(0.12)
        if klines_df is None:
            return None
        orderbook = None; social_score = 0; social_raw = {}
        try:
            ob_task = fetch_binance_order_book(session, pair_used, limit=100)
            sc_task = fetch_coingecko_social(session, cid)
            orderbook, (social_score, social_raw) = await asyncio.gather(ob_task, sc_task)
        except Exception:
            pass
        indicators = compute_technical_indicators(klines_df)
        if orderbook:
            obi, ob_liq = compute_orderbook_imbalance(orderbook, top_n=20)
            if obi is not None:
                indicators['orderbook_imbalance'] = obi
        price_change_24h = c.get('price_change_percentage_24h') or 0
        market_cap = c.get('market_cap') or 0
        try:
            recent = klines_df['close'].iloc[-4:].pct_change().dropna()
            momentum = float(recent.sum() * 100)
            base_score = max(0, min(40, momentum * 2 + max(min(price_change_24h, 40), 0) * 0.3))
        except Exception:
            base_score = max(0, min(40, price_change_24h * 0.6))
        score_info = compute_score(base_score, indicators, price_change_24h, market_cap, social_score=social_score)
        risk_flag = compute_risk(price_change_24h, c.get('total_volume',0), market_cap)
        vol_spike = indicators.get('vol_spike') or 0
        if vol_spike > 6 and indicators.get('orderbook_imbalance',0) > 4 and social_score < 2:
            risk_flag = True
        trade_levels = compute_trade_levels(c.get('current_price'), score_info['score'], indicators)
        model, feature_cols = load_model()
        ai_conf = None
        if model and feature_cols:
            ai_conf = model_predict_proba(model, feature_cols, {
                'score': score_info['score'],
                'indicators': indicators,
                'market_cap': market_cap,
                'current_price': c.get('current_price'),
                'price_change_24h': price_change_24h,
                'social_score': social_score
            })
        coin = {
            'name': c.get('name',''),
            'symbol': base_symbol,
            'pair': pair_used,
            'current_price': c.get('current_price'),
            'price_change_24h': price_change_24h,
            'score': score_info['score'],
            'reasons': score_info['reasons'],
            'coingecko_url': f"https://www.coingecko.com/en/coins/{cid}",
            'exchanges': [pair_used.split(base_symbol)[-1] + " (Binance)"] if pair_used else [],
            'risk_flag': risk_flag,
            'indicators': indicators,
            'social_raw': social_raw,
            'trade_levels': trade_levels,
            'social_score': social_score,
            'market_cap': market_cap,
            'ai_confidence': ai_conf
        }
        return coin

# ---------------- commands (Discord) ----------------
@bot.command(name='go')
async def cmd_start(ctx):
    if not poll_market.is_running():
        poll_market.start()
        auto_train_task.start()
        await ctx.send(f"Started polling every {POLL_INTERVAL_MIN} minutes and auto-train every {TRAIN_EVERY_HOURS} hours.")
    else:
        await ctx.send("Polling already running.")

@bot.command(name='stop')
async def cmd_stop(ctx):
    if poll_market.is_running():
        poll_market.stop()
        auto_train_task.stop()
        await ctx.send("Stopped polling and auto-training.")
    else:
        await ctx.send("Polling is not running.")

@bot.command(name='status')
async def cmd_status(ctx):
    model, feature_cols = load_model()
    model_status = "loaded" if model else "no model"
    await ctx.send(f"Bot running. Watchlist size: {len(watchlist)}. Model: {model_status}.")

@bot.command(name='watchlist')
async def cmd_watchlist(ctx):
    if not watchlist:
        await ctx.send("Watchlist empty.")
        return
    msg = "**Current Watchlist:**\n"
    for sym, data in watchlist.items():
        msg += f"{sym}: Score {data.get('score',0):.1f}, Last ${data.get('last_price')} (risk={data.get('risk_flag')}, ai={data.get('ai_confidence')})\n"
    await ctx.send(msg)

@bot.command(name='levels')
async def levels_cmd(ctx, symbol: str = None):
    if not symbol:
        await ctx.send("Usage: `!levels SYMBOL`")
        return
    sym = symbol.upper()
    entry = watchlist.get(sym)
    if not entry:
        await ctx.send(f"No watchlist entry found for {sym}. Try `!check {sym}` to run a live check.")
        return
    tl = entry.get('trade_levels')
    if not tl:
        await ctx.send(f"No trade levels stored for {sym}.")
        return
    out = f"**Levels for {sym}:**\n"
    if tl.get('entry'):
        out += f"Entry: ${tl['entry']:.6f}\n" if tl['entry'] < 1 else f"Entry: ${tl['entry']:.4f}\n"
    if tl.get('tp1'):
        out += f"TP1: ${tl['tp1']:.6f}\n" if tl['tp1'] < 1 else f"TP1: ${tl['tp1']:.4f}\n"
    if tl.get('tp2'):
        out += f"TP2: ${tl['tp2']:.6f}\n" if tl['tp2'] < 1 else f"TP2: ${tl['tp2']:.4f}\n"
    if tl.get('tp3'):
        out += f"TP3: ${tl['tp3']:.6f}\n" if tl['tp3'] < 1 else f"TP3: ${tl['tp3']:.4f}\n"
    if tl.get('sl'):
        out += f"SL: ${tl['sl']:.6f}\n" if tl['sl'] < 1 else f"SL: ${tl['sl']:.4f}\n"
    if tl.get('tl_pct') is not None:
        out += f"Trailing: {tl['tl_pct']*100:.2f}% (activate @ ${tl['tl_activate_at']:.4f})\n"
    if tl.get('notes'):
        out += f"Notes: {tl['notes']}\n"
    await ctx.send(out)

@bot.command(name='check')
async def check_cmd(ctx, symbol: str = None):
    if not symbol:
        await ctx.send("Usage: `!check SYMBOL`")
        return
    msg = await ctx.send(f"Running live check for {symbol.upper()}...")
    try:
        coin = await live_check_symbol(symbol)
        if not coin:
            await msg.edit(content=f"Could not find data for symbol {symbol.upper()}.")
            return
        embed = create_embed_for_coin(coin)
        await msg.delete()
        await ctx.send(embed=embed)
        watchlist[coin['symbol']] = {
            'score': coin['score'],
            'last_price': coin.get('current_price'),
            'risk_flag': coin.get('risk_flag', False),
            'timestamp': iso_now(),
            'trade_levels': coin.get('trade_levels', {}),
            'ai_confidence': coin.get('ai_confidence')
        }
        save_watchlist(watchlist)
    except Exception as e:
        logging.debug(f"live check failed: {e}")
        await msg.edit(content=f"Error while checking {symbol.upper()}: {e}")

@bot.command(name='train-now')
async def train_now_cmd(ctx):
    await ctx.send("Starting training (may take a while)...")
    loop = asyncio.get_event_loop()
    def train_sync():
        m,f = train_model_from_csv()
        return m is not None
    ok = await loop.run_in_executor(None, train_sync)
    if ok:
        await ctx.send("Training finished and model saved.")
    else:
        await ctx.send("Training failed or no data available.")

# ---------------- bot events ----------------
@bot.event
async def on_ready():
    logging.info(f"Bot connected as {bot.user}")
    if not poll_market.is_running():
        poll_market.start()
    if not auto_train_task.is_running():
        auto_train_task.start()

# ---------------- helper to collect labeled datapoints (offline) ----------------
# This function demonstrates how to append labeled rows into DATA_FILE for training:
# it requires a historical pipeline to compute next_horizon_pct (future return).
# Simplified approach: when running analyze_once, we can store a record with current features + 'future' = NaN,
# and a separate offline job (not included here) can backfill future returns using historical klines.
# For simplicity we will append "online" pseudo-labels when the future horizon has passed (example below).

# ---------------- run ----------------
if __name__ == "__main__":
    try:
        logging.info("Starting Crypto AI Bot")
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        logging.info("Shutting down.")

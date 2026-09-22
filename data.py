import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


SEED = 42                 # makes the output reproducible
N_USERS = 3000            # normal users
N_AGENTS = 200            # cash-in / cash-out agents
N_FRAUD_USERS = 50        # planted fraud cohort
START = datetime(2025, 1, 1)
MONTHS = 18               # length of the dataset
 
# transaction type mix (must sum to 1)
TYPE_MIX = {"cash_in": 0.30, "p2p": 0.35, "merchant": 0.20, "cash_out": 0.15}
 
# failure rate per channel -- USSD is flakier than the app
FAIL_RATE = {"app": 0.015, "ussd": 0.060, "agent": 0.025}
 
# amount distribution (lognormal). median = exp(MU) BDT
AMOUNT_MU = 8.0
AMOUNT_SIGMA = 1.5
 
CHURN_PROB = 0.06         # chance an active user goes dormant each month
REACTIVATE_PROB = 0.15    # chance a dormant user comes back
 
OUT_DIR = "data"
 
rng = np.random.default_rng(SEED)
# ==================================================================
# HELPERS
# ==================================================================

def month_starts(start, n):
    """Return the first day of each of n consecutive months."""
    out = []
    y, m = start.year, start.month
    for _ in range(n):
        out.append(datetime(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def days_in_month(dt):
    nxt = datetime(dt.year + (dt.month == 12), (dt.month % 12) + 1, 1)
    return (nxt - dt).days


def day_weights(month_start):
    n = days_in_month(month_start)
    w = np.ones(n)
    w[0] = 4.0
    w[min(24, n - 1):min(28, n)] = 3.0
    for d in range(n):
        # Python: Monday=0 ... Friday=4
        if (month_start + timedelta(days=d)).weekday() == 4:
            w[d] *= 0.55
    return w / w.sum()


HOUR_W = np.array(
    [0.4, 0.3, 0.2, 0.2, 0.3, 0.6, 1.2, 2.0, 3.0, 4.0, 4.5, 4.5,
     4.0, 4.0, 4.5, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.6]
)
HOUR_W = HOUR_W / HOUR_W.sum()

FRAUD_HOUR_W = np.array(
    [3.0, 4.0, 4.0, 3.5, 3.0, 1.5, 0.8, 0.6, 0.6, 0.7, 0.7, 0.7,
     0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]
)
FRAUD_HOUR_W = FRAUD_HOUR_W / FRAUD_HOUR_W.sum()


def draw_amount(n=1):
    amt = rng.lognormal(AMOUNT_MU, AMOUNT_SIGMA, n)
    amt = np.clip(amt, 50, 500_000)
    snap = rng.random(n) < 0.40
    rounded = np.where(amt < 5000,
                       np.round(amt / 100) * 100,
                       np.round(amt / 500) * 500)
    amt = np.where(snap, rounded, np.round(amt, 2))
    return np.maximum(amt, 50)


def pick_agent(agent_ids, agent_w, n=1):
    return rng.choice(agent_ids, size=n, p=agent_w)


def pick_status(channel):
    return "failed" if rng.random() < FAIL_RATE[channel] else "success"


# ==================================================================
# STEP 1 -- BUILD THE USER TABLE FIRST
# ==================================================================
print("Step 1: building users...")

months = month_starts(START, MONTHS)
END = months[-1] + timedelta(days=days_in_month(months[-1]) - 1)

# more users join in later months -> growth
join_weights = np.linspace(1.0, 3.0, MONTHS)
join_weights /= join_weights.sum()
join_month_idx = rng.choice(MONTHS, size=N_USERS, p=join_weights)

users = pd.DataFrame({
    "user_id": [f"U{100000 + i}" for i in range(N_USERS)],
    "join_month_idx": join_month_idx,
    # most users are light, a few are very heavy -> lognormal
    "activity_level": np.clip(rng.lognormal(-0.05, 0.8, N_USERS), 0.5, 60),
    "preferred_channel": rng.choice(["app", "ussd", "agent"],
                                    size=N_USERS, p=[0.45, 0.35, 0.20]),
})
users["join_date"] = [months[i] for i in users["join_month_idx"]]

agent_ids = np.array([f"A{2000 + i}" for i in range(N_AGENTS)])
# a few agents are very busy, most are quiet (zipf-like)
agent_w = 1.0 / np.arange(1, N_AGENTS + 1) ** 0.7
agent_w /= agent_w.sum()
rng.shuffle(agent_w)

# ==================================================================
# STEP 2-5 -- GENERATE NORMAL TRANSACTIONS
# ==================================================================
print("Step 2-5: generating normal transactions...")

type_names = list(TYPE_MIX.keys())
type_probs = list(TYPE_MIX.values())

rows = []
growth = np.linspace(1.0, 2.2, MONTHS)   # platform grows over time

for u in users.itertuples(index=False):
    active = True
    for mi in range(u.join_month_idx, MONTHS):
        if not active:
            if rng.random() < REACTIVATE_PROB:
                active = True
            else:
                continue

        expected = u.activity_level * growth[mi]
        n_txn = rng.poisson(expected)

        if n_txn > 0:
            ms = months[mi]
            dw = day_weights(ms)
            days = rng.choice(len(dw), size=n_txn, p=dw)
            hours = rng.choice(24, size=n_txn, p=HOUR_W)
            mins = rng.integers(0, 60, n_txn)
            amts = draw_amount(n_txn)
            types = rng.choice(type_names, size=n_txn, p=type_probs)

            for k in range(n_txn):
                ttype = types[k]
                channel = (u.preferred_channel if rng.random() < 0.75
                           else rng.choice(["app", "ussd", "agent"]))
                if ttype in ("cash_in", "cash_out"):
                    channel = "agent" if rng.random() < 0.85 else channel
                    agent = pick_agent(agent_ids, agent_w, 1)[0]
                else:
                    agent = ""

                rows.append((
                    u.user_id,
                    ms + timedelta(days=int(days[k]),
                                   hours=int(hours[k]),
                                   minutes=int(mins[k])),
                    float(amts[k]), ttype, agent, channel,
                    pick_status(channel),
                ))

        # churn check
        if rng.random() < CHURN_PROB:
            active = False

normal = pd.DataFrame(rows, columns=[
    "user_id", "timestamp", "amount", "txn_type",
    "agent_id", "channel", "status"])

# ==================================================================
# STEP 6 -- THE FRAUD COHORT (generated separately, then appended)
# ==================================================================
print("Step 6: planting fraud cohort...")

fraud_rows = []
fraud_ids = [f"U{900000 + i}" for i in range(N_FRAUD_USERS)]
TRAITS = ["layering", "structuring", "velocity", "odd_hours", "agent_concentration"]

for fid in fraud_ids:
    # each fraud user shows only SOME traits -- never all of them
    traits = rng.choice(TRAITS, size=rng.integers(2, 5), replace=False)
    my_agents = pick_agent(agent_ids, agent_w, 3)

    start_mi = int(rng.integers(0, MONTHS - 2))
    n_bursts = int(rng.integers(1, 4))

    for _ in range(n_bursts):
        mi = min(start_mi + int(rng.integers(0, 3)), MONTHS - 1)
        ms = months[mi]
        day = int(rng.integers(0, days_in_month(ms)))
        hour = int(rng.choice(24, p=FRAUD_HOUR_W)) if "odd_hours" in traits \
            else int(rng.choice(24, p=HOUR_W))
        base_t = ms + timedelta(days=day, hours=hour,
                                minutes=int(rng.integers(0, 60)))

        burst = int(rng.integers(8, 16)) if "velocity" in traits \
            else int(rng.integers(2, 5))

        for j in range(burst):
            t = base_t + timedelta(minutes=int(rng.integers(1, 60)) * j // 2)
            amt = float(draw_amount(1)[0])
            if "structuring" in traits:
                # sit just under a round threshold
                thresh = float(rng.choice([10_000, 25_000, 50_000, 100_000]))
                amt = thresh - float(rng.integers(100, 800))

            agent = str(rng.choice(my_agents)) if "agent_concentration" in traits \
                else pick_agent(agent_ids, agent_w, 1)[0]

            fraud_rows.append((fid, t, round(amt, 2), "cash_in",
                               agent, "agent", "success"))

            if "layering" in traits:
                t2 = t + timedelta(minutes=int(rng.integers(2, 21)))
                out_amt = round(max(50.0, amt * float(rng.uniform(0.90, 0.99))), 2)
                fraud_rows.append((fid, t2, out_amt, "cash_out",
                                   agent, "agent", "success"))

fraud = pd.DataFrame(fraud_rows, columns=normal.columns)
print(f"   fraud rows: {len(fraud):,}  "
      f"({100 * len(fraud) / (len(normal) + len(fraud)):.2f}% of total)")

# ==================================================================
# STEP 7 -- ASSEMBLE AND EXPORT
# ==================================================================
print("Step 7: assembling...")

df = pd.concat([normal, fraud], ignore_index=True)
df = df.sort_values("timestamp").reset_index(drop=True)
# IDs assigned LAST so fraud rows are not clustered at the end
df.insert(0, "txn_id", [f"T{1_000_000 + i}" for i in range(len(df))])
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(f"{OUT_DIR}/transactions.csv", index=False)
pd.DataFrame({"user_id": fraud_ids}).to_csv(
    f"{OUT_DIR}/fraud_labels.csv", index=False)

# ==================================================================
# STEP 8 -- SANITY CHECKS  (read these before touching Power BI)
# ==================================================================
print("\n" + "=" * 58)
print("SANITY CHECKS")
print("=" * 58)

d = df.copy()
d["timestamp"] = pd.to_datetime(d["timestamp"])

print(f"\n1. Rows: {len(d):,} | Users: {d.user_id.nunique():,} | "
      f"Range: {d.timestamp.min().date()} -> {d.timestamp.max().date()}")

print("\n2. Transactions per month (should trend UP):")
per_month = d.groupby(d.timestamp.dt.to_period('M')).size()
for p, v in per_month.items():
    print(f"   {p}  {v:6,}  {'#' * int(v / 120)}")

print("\n3. Day-of-month spikes (1st and 25-28 should stand out):")
per_day = d.groupby(d.timestamp.dt.day).size()
top = per_day.sort_values(ascending=False).head(6)
print("   busiest days:", list(top.index))

print("\n4. Amounts:")
print(f"   min {d.amount.min():,.0f} | median {d.amount.median():,.0f} "
      f"| max {d.amount.max():,.0f}")

lead = d.amount.astype(str).str.lstrip("0.").str[0]
lead = lead[lead.str.isdigit() & (lead != "0")]
obs = lead.value_counts(normalize=True).sort_index()
print("\n   Leading digit  observed  Benford")
for i in range(1, 10):
    exp = np.log10(1 + 1 / i)
    o = obs.get(str(i), 0)
    print(f"        {i}         {o:6.3f}    {exp:6.3f}")

print("\n5. Failure rate by channel (USSD should be worst):")
print((d.status.eq("failed").groupby(d.channel).mean() * 100)
      .round(2).to_string())

print("\n6. Cohort sizes (users by join month):")
first = d.groupby("user_id").timestamp.min().dt.to_period("M")
print(first.value_counts().sort_index().to_string())

print("\nDone. Files written to ./data/")
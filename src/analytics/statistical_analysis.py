import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')


def load_config():
    config_path = Path("F:/predictive-maintenance-digital-twin/configs/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_engine():
    config = load_config()['database']
    connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['name']}"
    return create_engine(connection_string)


def get_drive_lifetimes():
    query = """
    WITH drive_history AS (
        SELECT 
            serial_number,
            model,
            MIN(date) as first_seen,
            MAX(date) as last_seen,
            MAX(failure) as failed
        FROM hard_drive.smart_data
        GROUP BY serial_number, model
    )
    SELECT 
        serial_number,
        model,
        first_seen,
        last_seen,
        (last_seen - first_seen) + 1 as lifetime_days,
        failed
    FROM drive_history
    WHERE (last_seen - first_seen) >= 0
    """
    engine = get_engine()
    return pd.read_sql(query, engine)


class WeibullAnalysis:
    
    def __init__(self, lifetimes, censored):
        self.lifetimes = np.array(lifetimes, dtype=float)
        self.censored = np.array(censored, dtype=bool)
        self.failed = ~self.censored
        
        valid = self.lifetimes > 0
        self.lifetimes = self.lifetimes[valid]
        self.censored = self.censored[valid]
        self.failed = self.failed[valid]
        
        self.shape = None
        self.scale = None
        
    def fit(self):
        
        def neg_log_likelihood(params):
            beta, eta = params
            if beta <= 0 or eta <= 0:
                return np.inf
            
            ll = 0
            for t, is_censored in zip(self.lifetimes, self.censored):
                if is_censored:
                    ll += -((t / eta) ** beta)
                else:
                    ll += np.log(beta) - np.log(eta) + (beta - 1) * np.log(t / eta) - (t / eta) ** beta
            
            return -ll
        
        result = minimize(neg_log_likelihood, x0=[1.5, np.median(self.lifetimes)],
                         bounds=[(0.1, 10), (1, 10000)], method='L-BFGS-B')
        
        self.shape = result.x[0]
        self.scale = result.x[1]
        
        return self
    
    def reliability(self, t):
        return np.exp(-((t / self.scale) ** self.shape))
    
    def hazard_rate(self, t):
        return (self.shape / self.scale) * ((t / self.scale) ** (self.shape - 1))
    
    def mttf(self):
        from scipy.special import gamma
        return self.scale * gamma(1 + 1/self.shape)
    
    def b_life(self, percentile=10):
        return self.scale * ((-np.log(1 - percentile/100)) ** (1/self.shape))
    
    def summary(self):
        print("\nWEIBULL RELIABILITY ANALYSIS")
        print("-" * 60)
        print(f"\nDataset:")
        print(f"  Total drives:    {len(self.lifetimes):,}")
        print(f"  Failed drives:   {sum(self.failed):,}")
        print(f"  Censored drives: {sum(self.censored):,}")
        
        print(f"\nWeibull Parameters:")
        print(f"  Shape (beta):  {self.shape:.4f}")
        print(f"  Scale (eta):   {self.scale:.2f} days")
        
        print(f"\nInterpretation:")
        if self.shape < 1:
            print("  beta < 1: INFANT MORTALITY pattern")
            print("  -> Failure rate decreases over time")
            print("  -> Focus on burn-in testing")
        elif self.shape == 1:
            print("  beta = 1: RANDOM FAILURES pattern")
            print("  -> Constant failure rate")
        else:
            print("  beta > 1: WEAR-OUT FAILURES pattern")
            print("  -> Failure rate increases over time")
            print("  -> Preventive maintenance recommended")
        
        print(f"\nReliability Metrics:")
        print(f"  MTTF (Mean Time To Failure): {self.mttf():.1f} days")
        print(f"  B10 Life (10% failure):      {self.b_life(10):.1f} days")
        print(f"  B50 Life (50% failure):      {self.b_life(50):.1f} days")
        
        print(f"\nReliability at key timepoints:")
        for days in [30, 90, 180, 365]:
            r = self.reliability(days)
            print(f"  R({days} days): {r*100:.2f}%")


class KaplanMeierAnalysis:
    
    def __init__(self, lifetimes, censored):
        self.lifetimes = np.array(lifetimes)
        self.censored = np.array(censored)
        self.survival_times = None
        self.survival_probs = None
        
    def fit(self):
        event_times = np.unique(self.lifetimes[~self.censored])
        event_times = np.sort(event_times)
        
        survival_probs = []
        current_prob = 1.0
        
        n_at_risk = len(self.lifetimes)
        
        for t in event_times:
            d_t = np.sum((self.lifetimes == t) & (~self.censored))
            
            censored_before = np.sum((self.lifetimes < t) & (self.censored))
            
            n_at_risk = len(self.lifetimes) - np.sum(self.lifetimes < t)
            
            if n_at_risk > 0:
                current_prob *= (1 - d_t / n_at_risk)
            
            survival_probs.append(current_prob)
        
        self.survival_times = event_times
        self.survival_probs = np.array(survival_probs)
        
        return self
    
    def median_survival(self):
        if self.survival_probs is None:
            self.fit()
        
        idx = np.where(self.survival_probs <= 0.5)[0]
        if len(idx) > 0:
            return self.survival_times[idx[0]]
        return None


def compare_failure_rates(model1_data, model2_data):
    contingency = np.array([
        [sum(model1_data['failed']), sum(~model1_data['failed'])],
        [sum(model2_data['failed']), sum(~model2_data['failed'])]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def run_statistical_analysis():
    
    print("Loading drive lifetime data...")
    df = get_drive_lifetimes()
    print(f"Loaded {len(df):,} drives")
    
    print("\nSECTION 1: WEIBULL ANALYSIS (ALL DRIVES)")
    print("-" * 60)
    
    weibull = WeibullAnalysis(
        lifetimes=df['lifetime_days'].values,
        censored=(df['failed'] == 0).values
    )
    weibull.fit()
    weibull.summary()
    
    print("\nSECTION 2: WEIBULL BY DRIVE MODEL")
    print("-" * 60)
    
    top_models = df.groupby('model').size().nlargest(5).index.tolist()
    
    for model in top_models:
        model_df = df[df['model'] == model]
        if len(model_df) < 100:
            continue
            
        print(f"\n--- {model} ({len(model_df):,} drives) ---")
        
        w = WeibullAnalysis(
            lifetimes=model_df['lifetime_days'].values,
            censored=(model_df['failed'] == 0).values
        )
        w.fit()
        
        failure_count = model_df['failed'].sum()
        print(f"  Failures: {failure_count}")
        print(f"  Shape (beta): {w.shape:.3f}")
        print(f"  Scale (eta):  {w.scale:.1f} days")
        print(f"  MTTF:         {w.mttf():.1f} days")
    
    print("\nSECTION 3: KAPLAN-MEIER SURVIVAL ANALYSIS")
    print("-" * 60)
    
    km = KaplanMeierAnalysis(
        lifetimes=df['lifetime_days'].values,
        censored=(df['failed'] == 0).values
    )
    km.fit()
    
    print(f"\nSurvival Probabilities:")
    for days in [30, 60, 90, 180, 270]:
        idx = np.searchsorted(km.survival_times, days)
        if idx < len(km.survival_probs):
            print(f"  S({days} days): {km.survival_probs[idx]*100:.2f}%")
    
    median = km.median_survival()
    if median:
        print(f"\nMedian Survival Time: {median:.1f} days")
    else:
        print("\nMedian Survival Time: Not reached (>50% still operating)")
    


if __name__ == "__main__":
    run_statistical_analysis()

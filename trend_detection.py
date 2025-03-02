import pandas as pd
import numpy as np
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from scipy.stats import norm, kendalltau  # Core dependency
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class TrendDetector:
    """Main trend detection class with enhanced visualization"""
    
    class MKDetector:
        def __init__(
            self,
            data: pd.DataFrame,
            threshold: float = 0.8,
            alpha: float = 0.05,
            window_size: int = 20,
            direction: str = "both",
            ema_window: Optional[int] = None
        ):
            self.data = data.copy()
            self.threshold = threshold
            self.alpha = alpha
            self.window_size = window_size
            self.direction = direction
            self.ema_window = ema_window
            
            # Initialize price_column FIRST
            if self.ema_window is not None:
                self._add_ema_column(ema_window)
                self.price_column = f'adjclose_ema_{ema_window}'
            else:
                self.price_column = 'adjclose'
                
            # THEN validate inputs
            self._validate_inputs()

        
        def _validate_inputs(self):
            required_columns = [self.price_column, 'adjclose']
            missing = [col for col in required_columns if col not in self.data.columns]
            if missing:
                raise ValueError(f"MKDetector: Missing required columns {missing}")
            if not isinstance(self.data.index, pd.DatetimeIndex):
                raise ValueError("Data index must be a DatetimeIndex")

        @staticmethod
        def add_moving_average(data: pd.DataFrame, window: int) -> pd.DataFrame:
            """Adds exponential moving average column to dataframe"""
            df = data.copy()
            df = df.sort_index(ascending=True)
            ema_col = f'adjclose_ema_{window}'
            df[ema_col] = df['adjclose'].ewm(span=window, adjust=False).mean()
            return df

        def _add_ema_column(self, window: int):
            """Internal method to add EMA column to data"""
            self.data = self.add_moving_average(self.data, window)

        def detect_trends(self) -> pd.DataFrame:
            """Detects trends using either raw prices or EMA"""
            results = []
            prices = self.data[self.price_column].sort_index(ascending=True)
            
            for i in range(len(prices)):
                if i < self.window_size - 1:
                    continue
                
                window = prices.iloc[i - self.window_size + 1 : i + 1]
                end_date = window.index[-1]
                
                if len(window) < self.window_size:
                    continue
                    
                try:
                    z_stat, p_value = self._calculate_mk_test(window.values)
                except ValueError as e:
                    raise ValueError(f"detect_trends: {str(e)} at {end_date}") from e
                
                direction = 'up' if z_stat > 0 else ('down' if z_stat < 0 else 'flat')
                if self.direction != "both" and direction != self.direction:
                    continue
                    
                results.append({
                    'date': end_date,
                    'price': window.iloc[-1],
                    'trend_strength': abs(z_stat),
                    'direction': direction,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                })
            
            return pd.DataFrame(results).set_index('date')

        
        def _calculate_mk_test(self, prices: np.ndarray) -> tuple:
            """
            Performs the Mann–Kendall test.
            
            Returns:
                z_stat (float): The raw test statistic (preserving sign).
                p (float): The two-tailed p-value.
            """
            n = len(prices)
            # Compute the Mann–Kendall statistic s
            s = sum(np.sign(prices[j] - prices[i]) for i in range(n - 1) for j in range(i + 1, n))
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            if var_s == 0:
                raise ValueError("Variance of S is zero; cannot compute test statistic")
                
            # Compute z_stat keeping the sign to determine trend direction
            z_stat = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0
            
            # Calculate two-tailed p-value
            p = 2 * (1 - norm.cdf(abs(z_stat)))
            
            return z_stat, p

        
        def plot_trends(self, results: pd.DataFrame, symbol: str = 'Stock') -> None:
            """Enhanced plotting with trend segments and EMA comparison"""
            fig, ax = plt.subplots(figsize=(15, 7))
            
            # Plot raw prices if using EMA analysis
            if self.ema_window is not None:
                ax.plot(self.data.index, self.data['adjclose'], 
                        color='#CCCCCC', alpha=0.3, linewidth=1, 
                        label='Raw Price')
            
            # Plot analysis prices (EMA or raw)
            ax.plot(self.data.index, self.data[self.price_column],
                    color='#1f77b4', alpha=0.3, 
                    label=f'{"EMA" if self.ema_window else "Price"} ({self.price_column})')

            # Plot significant trend segments
            sig_trends = results[results['significant']]
            prev_idx = None
            current_color = None
            
            for idx, (date, row) in enumerate(sig_trends.iterrows()):
                color = 'green' if row['direction'] == 'up' else 'red'
                
                if color != current_color or current_color is None:
                    if prev_idx is not None:
                        # Plot the previous trend segment
                        segment = sig_trends.iloc[prev_idx:idx]
                        ax.plot(segment.index, segment['price'],  # Changed to 'price' column
                                color=current_color, linewidth=1)
                    current_color = color
                    prev_idx = idx

                    # Annotate at trend changes
                    # ax.annotate(
                    #     f"Str: {row['trend_strength']:.2f}\np: {row['p_value']:.3f}",
                    #     xy=(date, row['price']),  # Changed to 'price' column
                    #     xytext=(10, -20),
                    #     textcoords='offset points',
                    #     color=color,
                    #     fontsize=8,
                    #     arrowprops=dict(
                    #         arrowstyle="->", 
                    #         color=color,
                    #         alpha=0.7
                    #     )
                    # )

            # Plot final segment
            if prev_idx is not None:
                segment = sig_trends.iloc[prev_idx:]
                ax.plot(segment.index, segment['price'],  # Changed to 'price' column
                        color=current_color, linewidth=1)

            ax.set_title(f'{symbol} Price with Trend Detection')
            ax.set_ylabel('Adjusted Close')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()





ticker = 'QQQ'
# Updated usage example
if __name__ == "__main__":
    from fmp_helpers import FMPHelpers
    # Generate sample data with clear trend

    fmp = FMPHelpers(api_key=api, project_id=PROJECT_ID)
    data = fmp.get_daily_price_volume(ticker, '2018-01-01', '2025-02-22')
    data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    # Detect trends
    detector_50 = TrendDetector.MKDetector(
        data=data,
        window_size=10,  # 2-week lookback
        alpha=0.05,
        ema_window=50
        # price_column='adjclose_ema_10'
    )
    detector_10 = TrendDetector.MKDetector(
        data=data,
        window_size=10,  # 2-week lookback
        alpha=0.05,
        ema_window=10
        # price_column='adjclose_ema_10'
    )
    # data_ema = detector.add_moving_average(data, 10)
    trends_50 = detector_50.detect_trends()
    trends_10 = detector_10.detect_trends()
    
    # Plot results
    # detector_50.plot_trends(trends_50, symbol=ticker)
    detector_10.plot_trends(trends_10, symbol=ticker)
    
    
    # # Show top 5 strongest trends
    # print("\nStrongest Detected Trends:")
    # print(trends.sort_values('trend_strength', ascending=False).head())

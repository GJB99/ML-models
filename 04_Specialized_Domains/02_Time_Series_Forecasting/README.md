# 02_Time_Series_Forecasting

Time series forecasting is a critical task in many industries, involving the prediction of future values based on historical, time-ordered data. This can include anything from stock prices and weather patterns to sales data and server demand.

### Classical Statistical Models:

-   **ARIMA (AutoRegressive Integrated Moving Average)**: A widely used statistical model for analyzing and forecasting time series data. It combines three components:
    -   **AR (AutoRegression)**: A model that uses the dependent relationship between an observation and some number of lagged observations.
    -   **I (Integrated)**: The use of differencing of raw observations (e.g., subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
    -   **MA (Moving Average)**: A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
-   **Prophet**: A forecasting tool developed by Facebook, designed to handle time series data with strong seasonal effects and several seasons of historical data. It is robust to missing data and shifts in the trend, and typically handles outliers well.

### Deep Learning Approaches:

-   **RNNs/LSTMs**: Recurrent Neural Networks are naturally suited for sequential data like time series.
-   **TCNs (Temporal Convolutional Networks)**: Often outperform RNNs on time series tasks due to their stable gradients and large receptive fields.
-   **Transformer-based Models**: More recently, Transformer architectures have been adapted for time series forecasting, showing strong performance in capturing long-range dependencies. 
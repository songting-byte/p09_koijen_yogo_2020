## Chart: Sectoral Debt Securities Positions (BIS DSS)

**Description:**  
This chart plots the annual average outstanding positions (`OBS_VALUE`) of debt securities using BIS Debt Securities Statistics. Values are aggregated by calendar year and institutional sector, capturing the evolution of balance sheet debt exposures across sectors over time. The data represent **stocks (closing balance sheet positions)** rather than flows, and therefore reflect structural funding conditions and long-term leverage dynamics in the financial system rather than short-term market movements.

**Economic Interpretation:**  
Debt securities positions measure the scale of long-term market-based financing used by different sectors. Increases in these positions indicate rising reliance on capital market funding and higher leverage embedded in balance sheets, while contractions reflect deleveraging, fiscal consolidation, or reduced market access. Sectoral divergence in debt growth highlights asymmetries in financial vulnerability and exposure to refinancing risk.

**Relevance for Financial Stability:**  
Large and growing stocks of debt securities amplify systemic risk through rollover exposure, refinancing fragility, and interest-rate sensitivity. Concentration of debt in specific sectors increases interconnectedness and contagion risk, especially during liquidity tightening cycles. This chart provides a structural view of macro-financial risk accumulation that complements short-term stress indicators and market-based risk measures.


**Transformations and Aggregation:**  
Observations are:
- Aggregated by calendar year and institutional sector  
- Averaged across observations to construct annual sectoral series  

**Formulas Used:**

```{math}
\begin{align*}
year\_annual &= \text{YEAR}(TIME\_PERIOD) \\
avg\_value_{t,s} &= \frac{1}{N_{t,s}} \sum_{i=1}^{N_{t,s}} OBS\_VALUE_{i,t,s}
\end{align*}

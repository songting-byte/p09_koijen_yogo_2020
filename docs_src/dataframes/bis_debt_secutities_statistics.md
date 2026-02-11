
## Description

This data set covers borrowing and investment activities in debt capital markets, capturing debt instruments designed to be traded in financial markets such as treasury bills, commercial paper, negotiable certificates of deposit, bonds, debentures and asset-backed securities.Total debt securities are issued by residents in all markets. Domestic (international) debt securities are issued in (outside) the local market of the country where the borrower resides, regardless of the currency denomination of the security. As valuation methods differ across countries, some amounts are presented at market value and others at nominal or face value.

---
# Dataset Description


- **FREQ**: `object`  
  Data frequency (e.g. Annual)

- **ADJUSTMENT**: `object`  
  Seasonal / calendar adjustment indicator

- **REF_AREA**: `object`  
  Reporting country / economy (ISO code)

- **COUNTERPART_AREA**: `object`  
  Counterpart country / economy (e.g. World)

- **REF_SECTOR**: `object`  
  Reporting institutional sector (e.g. S13 = General government)

- **COUNTERPART_SECTOR**: `object`  
  Counterpart institutional sector (e.g. S1 = Total economy)

- **CONSOLIDATION**: `object`  
  Consolidation status (e.g. non-consolidated)

- **ACCOUNTING_ENTRY**: `object`  
  Accounting entry (assets or liabilities)

- **STO**: `object`  
  Stocks / transactions / other flows indicator  
  (e.g. closing balance sheet positions)

- **INSTR_ASSET**: `object`  
  Financial instrument / asset classification  
  (e.g. F3 = Debt securities)

- **MATURITY**: `object`  
  Original / residual maturity classification  
  (e.g. long-term maturity)

- **UNIT_MEASURE**: `object`  
  Unit of measure (e.g. USD)

- **CURRENCY**: `object`  
  Currency of denomination  
  (e.g. domestic vs non-domestic currency)

- **VALUATION**: `object`  
  Valuation method (e.g. nominal value)

- **PRICES**: `object`  
  Price basis (e.g. current prices)

- **TRANSFORMATION**: `object`  
  Data transformation indicator


- **TIME_PERIOD**: `object`  
  Observation year

- **OBS_VALUE**: `float64`  
  Observed value of debt securities position





---

## Source

Bank for International Settlements (BIS)  
Debt Securities Statistics (DSS)  
API: BIS SDMX v2  
Provider: `stats.bis.org`


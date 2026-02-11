# Dataframe: `P0:bis` - BIS Debt securities statistics


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




## DataFrame Glimpse

```
Rows: 3809
Columns: 19
$ FREQ               <str> 'A'
$ ADJUSTMENT         <str> 'N'
$ REF_AREA           <str> 'AU'
$ COUNTERPART_AREA   <str> 'XW'
$ REF_SECTOR         <str> 'S13'
$ COUNTERPART_SECTOR <str> 'S1'
$ CONSOLIDATION      <str> 'N'
$ ACCOUNTING_ENTRY   <str> 'L'
$ STO                <str> 'LE'
$ INSTR_ASSET        <str> 'F3'
$ MATURITY           <str> 'L'
$ EXPENDITURE        <str> '_Z'
$ UNIT_MEASURE       <str> 'USD'
$ CURRENCY_DENOM     <str> 'XDC'
$ VALUATION          <str> 'N'
$ PRICES             <str> 'V'
$ TRANSFORMATION     <str> 'N'
$ TIME_PERIOD        <str> '2020'
$ OBS_VALUE          <f64> 866.6578664


```

## Dataframe Manifest

| Dataframe Name                 | BIS Debt securities statistics                                                   |
|--------------------------------|--------------------------------------------------------------------------------------|
| Dataframe ID                   | [bis](../dataframes/P0/bis.md)                                       |
| Data Sources                   | National data, BIS calculations                                        |
| Data Providers                 | National data, BIS calculations                                      |
| Links to Providers             | https://data.bis.org/topics/DSS/data?data_view=table                             |
| Topic Tags                     | Debt Security, Repo                                          |
| Type of Data Access            | P,u,b,l,i,c                                  |
| How is data pulled?            | Web API via Python and Pandas                                                    |
| Data available up to (min)     | None                                                             |
| Data available up to (max)     | None                                                             |
| Dataframe Path                 | /Users/nandinikrishnan/Documents/GitHub/full_stack_quant_finance/p09_koijen_yogo_2020/_data/bis_debt_securities_cleaned.parquet                                                   |


**Linked Charts:**


- [P0:bis_chart](../../charts/P0.bis_chart.md)



## Pipeline Manifest

| Pipeline Name                   | p09_koijen_yogo_2020                       |
|---------------------------------|--------------------------------------------------------|
| Pipeline ID                     | [P0](../index.md)              |
| Lead Pipeline Developer         | Allen Wu & Nandini Krishnan & Xiongfei Wang             |
| Contributors                    | Allen Wu & Nandini Krishnan & Xiongfei Wang           |
| Git Repo URL                    |                         |
| Pipeline Web Page               | <a href="file:///Users/nandinikrishnan/Documents/GitHub/full_stack_quant_finance/p09_koijen_yogo_2020/docs/index.html">Pipeline Web Page      |
| Date of Last Code Update        | 2026-02-10 22:26:58           |
| OS Compatibility                |  |
| Linked Dataframes               |  [P0:bis](../dataframes/P0/bis.md)<br>  |



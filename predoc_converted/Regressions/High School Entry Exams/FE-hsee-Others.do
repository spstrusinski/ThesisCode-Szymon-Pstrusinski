clear all
set more off
version 16
* 1. Load data
import excel using "C:\Users\Pstrrr...ICYK\Desktop\Teza\SzymonPstrusiński-Codes", firstrow clear


* 2. Key indicators
* Gymnasium indicator
gen byte SchoolTypeB = (SchoolType == "Gymasium")

* Urban gmina indicator
gen byte Urban = (gmina_class == "urban")


* 3. Panel construction

encode powiat_special, gen(powiat_id)

* Collapse to powiat-year means
collapse (mean) PolishAverage MathAverage EnglishAverage ///
                PolishTestTakers Urban SchoolTypeB, ///
         by(powiat_id T)

xtset powiat_id T


* 4. Fixed Effects regressions – full sample


* Math
xtreg MathAverage SchoolTypeB i.T, fe vce(cluster powiat_id)
est store MM1
xtreg MathAverage c.SchoolTypeB##c.Urban i.T, fe vce(cluster powiat_id)
est store MM2

* English
xtreg EnglishAverage SchoolTypeB i.T, fe vce(cluster powiat_id)
est store EM1
xtreg EnglishAverage c.SchoolTypeB##c.Urban i.T, fe vce(cluster powiat_id)
est store EM2

* Polish
xtreg PolishAverage SchoolTypeB i.T, fe vce(cluster powiat_id)
est store PM1
xtreg PolishAverage c.SchoolTypeB##c.Urban i.T, fe vce(cluster powiat_id)
est store PM2

* Test takers
xtreg PolishTestTakers SchoolTypeB i.T, fe vce(cluster powiat_id)
est store TM1
xtreg PolishTestTakers c.SchoolTypeB##c.Urban i.T, fe vce(cluster powiat_id)
est store TM2

estimates table MM1 MM2 EM1 EM2 PM1 PM2 TM1 TM2, stats(N) se p


* 5. Robustness check: exclude COVID years

preserve
drop if T >= 7 & T <= 9

* Repeat FE specifications
xtreg MathAverage SchoolTypeB i.T, fe vce(cluster powiat_id)
est store MM1_nc
xtreg MathAverage c.SchoolTypeB##c.Urban i.T, fe vce(cluster powiat_id)
est store MM2_nc

xtreg EnglishAverage SchoolTypeB i.T, fe vce(cluster powiat_id)
est store EM1_nc
xtreg EnglishAverage c.SchoolTypeB##c.Urban i.T, fe vce(cluster powiat_id)
est store EM2_nc

xtreg PolishAverage SchoolTypeB i.T, fe vce(cluster powiat_id)
est store PM1_nc
xtreg PolishAverage c.SchoolTypeB##c.Urban i.T, fe vce(cluster powiat_id)
est store PM2_nc

xtreg PolishTestTakers SchoolTypeB i.T, fe vce(cluster powiat_id)
est store TM1_nc
xtreg PolishTestTakers c.SchoolTypeB##c.Urban i.T, fe vce(cluster powiat_id)
est store TM2_nc

estimates table MM1_nc MM2_nc EM1_nc EM2_nc PM1_nc PM2_nc TM1_nc TM2_nc, stats(N) se p
restore

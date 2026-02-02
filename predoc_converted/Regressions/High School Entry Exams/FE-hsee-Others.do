* 2.  Generate dummies
gen byte SchoolTypeB = (SchoolType=="Gymasium")
gen byte Urban      = (gmina_class=="urban")

* 3.  Make sure T and powiat_special are set
encode powiat_special, gen(powiat_id)
collapse (mean) PolishAverage MathAverage EnglishAverage PolishTestTakers Urban SchoolTypeB, by(powiat_id T)

xtset powiat_id T

xtreg MathAverage SchoolTypeB i.T, fe robust cluster(powiat_id)
estimates store MM1
xtreg MathAverage c.SchoolTypeB##c.Urban i.T, fe robust cluster(powiat_id)
estimates store MM2
xtreg EnglishAverage SchoolTypeB i.T, fe robust cluster(powiat_id)
estimates store EM1
xtreg EnglishAverage c.SchoolTypeB##c.Urban i.T, fe robust cluster(powiat_id)
estimates store EM2
xtreg PolishAverage SchoolTypeB i.T, fe robust cluster(powiat_id)
estimates store PM1
xtreg PolishAverage c.SchoolTypeB##c.Urban i.T, fe robust cluster(powiat_id)
estimates store PM2
xtreg PolishTestTakers SchoolTypeB i.T, fe robust cluster(powiat_id)
estimates store TM1
xtreg PolishTestTakers c.SchoolTypeB##c.Urban i.T, fe robust cluster(powiat_id)
estimates store TM2

estimates table MM1 MM2 EM1 EM2 PM1 PM2 TM1 TM2, stats(N) se p


*Now, no COVID

drop if T>=7 & T<=9

xtreg MathAverage SchoolTypeB i.T, fe robust cluster(powiat_id)
estimates store MM1
xtreg MathAverage c.SchoolTypeB##c.Urban i.T, fe robust cluster(powiat_id)
estimates store MM2
xtreg EnglishAverage SchoolTypeB i.T, fe robust cluster(powiat_id)
estimates store EM1
xtreg EnglishAverage c.SchoolTypeB##c.Urban i.T, fe robust cluster(powiat_id)
estimates store EM2
xtreg PolishAverage SchoolTypeB i.T, fe robust cluster(powiat_id)
estimates store PM1
xtreg PolishAverage c.SchoolTypeB##c.Urban i.T, fe robust cluster(powiat_id)
estimates store PM2
xtreg PolishTestTakers SchoolTypeB i.T, fe robust cluster(powiat_id)
estimates store TM1
xtreg PolishTestTakers c.SchoolTypeB##c.Urban i.T, fe robust cluster(powiat_id)
estimates store TM2
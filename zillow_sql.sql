## Specific query with square footage, bedroom, and bathroom counts, house_value
SELECT prop.parcelid, calculatedfinishedsquarefeet, bathroomcnt, bedroomcnt,taxvaluedollarcnt
FROM properties_2017 as prop
JOIN predictions_2017 as pred
	ON prop.parcelid = pred.parcelid
	AND	(transactiondate like "2017-05%" or transactiondate like "2017-06%")
JOIN propertylandusetype as usetype
	ON prop.propertylandusetypeid = usetype.propertylandusetypeid
	AND prop.propertylandusetypeid in (261,262,263,264,265,268,269,273,275,276)

## All single unit properties with last transaction date between May and June 2017
SELECT count(*)
FROM properties_2017 as prop
JOIN predictions_2017 as pred
	ON prop.parcelid = pred.parcelid
	AND	(transactiondate like "2017-05%" or transactiondate like "2017-06%")
JOIN propertylandusetype as usetype
	ON prop.propertylandusetypeid = usetype.propertylandusetypeid
	AND prop.propertylandusetypeid in (261,262,263,264,265,268,269,273,275,276)
LIMIT 200

#Checking for types of house
SELECT *
FROM properties_2017 as prop
JOIN predictions_2017 as pred
	ON prop.parcelid = pred.parcelid
	AND	(transactiondate like "2017-05%" or transactiondate like "2017-06%")
JOIN propertylandusetype as usetype
	ON prop.propertylandusetypeid = usetype.propertylandusetypeid
	AND prop.propertylandusetypeid in (270)
LIMIT 200

#Checking for transaction date
SELECT count(*)
FROM properties_2017 as prop
JOIN predictions_2017 as pred
	ON prop.parcelid = pred.parcelid
	AND	(transactiondate like "2017-05%" or transactiondate like "2017-06%")
LIMIT 200

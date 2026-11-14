FetalAI demo samples (real dataset images; each gives a different result)
=========================================================================

Start the app with start_fetalai.bat, open "Scan", drop an image, press Analyze.
For HC18 images the pixel spacing fills in automatically (matched by file content).

File                                  What to enter            Expected result
------------------------------------  -----------------------  ---------------------------------------------
1_PASS_544_HC.png                     GA 20 weeks + 0 days     Plane PASS, HC 170.9 +/- 1.6 mm,
                                                               ~42nd centile, reliability OK
2_WARN_withheld_006_HC.png            nothing                  Plane WARN, HC withheld;
                                                               tick "Measure anyway" -> HC 67.3 mm
3_PASS_low_reliability_794_HC.png     nothing                  Plane PASS, HC 301.4 +/- 14.2 mm,
                                                               reliability "Low: re-scan"
4_WARN_femur_Patient00647.png         pixel spacing 0.1        Plane WARN: "fetal femur ... re-acquire"

CTG (no images exist in the UCI CTG dataset, only 21 measured numbers per record):
open "CTG" and click Normal / Suspect / Pathological, then "Classify CTG".
These load real held-out test records CTG0908, CTG0349 and CTG1955.

Then open "Report" for the combined screening summary.

Sources: HC18 (Zenodo 1327317, CC BY 4.0), FETAL_PLANES_DB (Zenodo 3904280, CC BY 4.0),
UCI Cardiotocography. Research prototype, not a medical device.

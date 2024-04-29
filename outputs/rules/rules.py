def findDecision(obj): #obj[0]: Pregnancies, obj[1]: Glucose, obj[2]: BloodPressure, obj[3]: SkinThickness, obj[4]: Insulin, obj[5]: BMI, obj[6]: DiabetesPedigreeFunction, obj[7]: Age
   # {"feature": "Glucose", "instances": 768, "metric_value": 0.0441, "depth": 1}
   if obj[1]<=120.89453125:
      # {"feature": "Age", "instances": 419, "metric_value": 0.024, "depth": 2}
      if obj[7]<=30.572792362768496:
         # {"feature": "BMI", "instances": 270, "metric_value": 0.0194, "depth": 3}
         if obj[5]<=29.5:
            # {"feature": "Pregnancies", "instances": 135, "metric_value": 0.0899, "depth": 4}
            if obj[0]<=6:
               # {"feature": "DiabetesPedigreeFunction", "instances": 130, "metric_value": 0.0346, "depth": 5}
               if obj[6]<=0.3902230769230769:
                  return 0
               elif obj[6]>0.3902230769230769:
                  return 0.020833333333333332
               else:
                  return 0.007692307692307693
            elif obj[0]>6:
               # {"feature": "DiabetesPedigreeFunction", "instances": 5, "metric_value": 0.4, "depth": 5}
               if obj[6]<=0.296:
                  return 1
               elif obj[6]>0.296:
                  return 0
               else:
                  return 0.8
            else:
               return 0.037037037037037035
         elif obj[5]>29.5:
            # {"feature": "DiabetesPedigreeFunction", "instances": 135, "metric_value": 0.0156, "depth": 4}
            if obj[6]<=0.6711554738243004:
               # {"feature": "Insulin", "instances": 112, "metric_value": 0.0306, "depth": 5}
               if obj[4]<=134.3638554324273:
                  return 0.13829787234042554
               elif obj[4]>134.3638554324273:
                  return 0
               else:
                  return 0.11607142857142858
            elif obj[6]>0.6711554738243004:
               # {"feature": "Insulin", "instances": 23, "metric_value": 0.0997, "depth": 5}
               if obj[4]<=125:
                  return 0.25
               elif obj[4]>125:
                  return 1
               else:
                  return 0.34782608695652173
            else:
               return 0.15555555555555556
         else:
            return 0.0962962962962963
      elif obj[7]>30.572792362768496:
         # {"feature": "BMI", "instances": 149, "metric_value": 0.0662, "depth": 3}
         if obj[5]>25.803467776871:
            # {"feature": "DiabetesPedigreeFunction", "instances": 122, "metric_value": 0.0098, "depth": 4}
            if obj[6]<=1.1191834479688851:
               # {"feature": "Pregnancies", "instances": 114, "metric_value": 0.0068, "depth": 5}
               if obj[0]<=6:
                  return 0.2923076923076923
               elif obj[0]>6:
                  return 0.4489795918367347
               else:
                  return 0.35964912280701755
            elif obj[6]>1.1191834479688851:
               # {"feature": "Pregnancies", "instances": 8, "metric_value": 0.433, "depth": 5}
               if obj[0]>2:
                  return 1
               elif obj[0]<=2:
                  return 0
               else:
                  return 0.75
            else:
               return 0.38524590163934425
         elif obj[5]<=25.803467776871:
            return 0
         else:
            return 0.31543624161073824
      else:
         return 0.17422434367541767
   elif obj[1]>120.89453125:
      # {"feature": "BMI", "instances": 349, "metric_value": 0.0137, "depth": 2}
      if obj[5]>26.552329380744702:
         # {"feature": "Age", "instances": 301, "metric_value": 0.0078, "depth": 3}
         if obj[7]>24.450888298876624:
            # {"feature": "DiabetesPedigreeFunction", "instances": 252, "metric_value": 0.0063, "depth": 4}
            if obj[6]<=0.5225873015873016:
               # {"feature": "Pregnancies", "instances": 154, "metric_value": 0.0083, "depth": 5}
               if obj[0]<=12:
                  return 0.5761589403973509
               elif obj[0]>12:
                  return 1
               else:
                  return 0.5844155844155844
            elif obj[6]>0.5225873015873016:
               # {"feature": "Pregnancies", "instances": 98, "metric_value": 0.0289, "depth": 5}
               if obj[0]<=7:
                  return 0.6571428571428571
               elif obj[0]>7:
                  return 0.9285714285714286
               else:
                  return 0.7346938775510204
            else:
               return 0.6428571428571429
         elif obj[7]<=24.450888298876624:
            # {"feature": "DiabetesPedigreeFunction", "instances": 49, "metric_value": 0.0252, "depth": 4}
            if obj[6]<=1.1855447093868852:
               # {"feature": "BloodPressure", "instances": 47, "metric_value": 0.0137, "depth": 5}
               if obj[2]>0:
                  return 0.3695652173913043
               elif obj[2]<=0:
                  return 1
               else:
                  return 0.3829787234042553
            elif obj[6]>1.1855447093868852:
               return 1
            else:
               return 0.40816326530612246
         else:
            return 0.6046511627906976
      elif obj[5]<=26.552329380744702:
         # {"feature": "Pregnancies", "instances": 48, "metric_value": 0.0717, "depth": 3}
         if obj[0]>2:
            # {"feature": "Age", "instances": 27, "metric_value": 0.0714, "depth": 4}
            if obj[7]<=60:
               # {"feature": "Insulin", "instances": 23, "metric_value": 0.0477, "depth": 5}
               if obj[4]<=175:
                  return 0.5714285714285714
               elif obj[4]>175:
                  return 0
               else:
                  return 0.5217391304347826
            elif obj[7]>60:
               return 0
            else:
               return 0.4444444444444444
         elif obj[0]<=2:
            # {"feature": "Age", "instances": 21, "metric_value": 0.1653, "depth": 4}
            if obj[7]<=29:
               return 0
            elif obj[7]>29:
               return 0.5
            else:
               return 0.047619047619047616
         else:
            return 0.2708333333333333
      else:
         return 0.5587392550143266
   else:
      return 0.3489583333333333

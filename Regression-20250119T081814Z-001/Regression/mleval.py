
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

def regval(xtrain, xtrain_poly, xtest, xtest_poly, ytrain, ytest, models):
    
    trainrmse = []
    testrmse = []
    trainr2 = []
    testr2 = []

    fit = []
    crossvalscore = []
    
    for name, model in models.items():
    
        if name!='poly':
            # RMSE, R2score
            ytrain_pred = model.predict(xtrain)
            ytest_pred = model.predict(xtest)

            trainrmse.append(round(np.sqrt(mean_squared_error(ytrain, ytrain_pred)),2))
            testrmse.append(round(np.sqrt(mean_squared_error(ytest, ytest_pred)),2))
            trainr2.append(round(r2_score(ytrain, ytrain_pred),2))
            testr2.append(round(r2_score(ytest, ytest_pred),2))
            trscore = r2_score(ytrain, ytrain_pred)
            tescore = r2_score(ytest, ytest_pred)

            # Bias-Variance Trade off
            if trscore>=0.60 and tescore>=0.60:
                if trscore>tescore:
                    if trscore-tescore>=0.10:
                        fit.append("Overfit")
                    elif trscore-tescore<0.10:
                        fit.append("Goodfit")
                    else:
                        fit.append('Nofit')
                elif trscore<tescore:
                    if tescore-trscore>=0.10:
                        fit.append("Overfit")
                    elif tescore-trscore<0.10:
                        fit.append("Goodfit")
                    else:
                        fit.append('Nofit')
                else:
                    fit.append("Nofit")

            elif trscore<0.60 and tescore<0.60:
                if abs(trscore)==0 and abs(tescore)==0:
                    fit.append("Nofit")
                else:
                    fit.append("Underfit")
            else:
                fit.append("Nofit")

            # Cross-val score
            crossX = pd.concat([xtrain, xtest], axis = 0)
            crossy = pd.concat([ytrain, ytest], axis = 0)

            scores = cross_val_score(models[name], crossX, crossy, cv=3) # Taking 3 folds
            crossvalscore.append(round(scores.mean(),2))
        else:
            # RMSE, R2score
            ytrain_pred = model.predict(xtrain_poly)
            ytest_pred = model.predict(xtest_poly)

            trainrmse.append(round(np.sqrt(mean_squared_error(ytrain, ytrain_pred)),2))
            testrmse.append(round(np.sqrt(mean_squared_error(ytest, ytest_pred)),2))
            trainr2.append(round(r2_score(ytrain, ytrain_pred),2))
            testr2.append(round(r2_score(ytest, ytest_pred),2))
            trscore = r2_score(ytrain, ytrain_pred)
            tescore = r2_score(ytest, ytest_pred)

            # Bias-Variance Trade off
            if trscore>=0.60 and tescore>=0.60:
                if trscore>tescore:
                    if trscore-tescore>=0.10:
                        fit.append("Overfit")
                    elif trscore-tescore<0.10:
                        fit.append("Goodfit")
                    else:
                        fit.append('Nofit')
                elif trscore<tescore:
                    if tescore-trscore>=0.10:
                        fit.append("Overfit")
                    elif tescore-trscore<0.10:
                        fit.append("Goodfit")
                    else:
                        fit.append('Nofit')
                else:
                    fit.append("Nofit")

            elif trscore<0.60 and tescore<0.60:
                if abs(trscore)==0 and abs(tescore)==0:
                    fit.append("Nofit")
                else:
                    fit.append("Underfit")
            else:
                fit.append("Nofit")

            # Cross-val score
            crossX = pd.concat([pd.DataFrame(xtrain_poly), pd.DataFrame(xtest_poly)], axis = 0)
            crossy = pd.concat([ytrain, ytest], axis = 0)

            scores = cross_val_score(models[name], crossX, crossy, cv=3) # Taking 3 folds
            crossvalscore.append(round(scores.mean(),2))
        
    return trainrmse, testrmse, trainr2, testr2, crossvalscore, fit
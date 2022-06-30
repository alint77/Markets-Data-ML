import numpy as np


def thread_function(df, threadidx, CandlesInDay,):

    takeProfitPoints = 1.002
    stopLoss = 0.998

    startCandleHour = 4
    startCandleIndex = int(startCandleHour * CandlesInDay) // 24

    windowLengthHour = 17
    windowLengthCount = int(windowLengthHour * CandlesInDay) // 24

    dataLengthHour = 3
    dataLengthCount = int(dataLengthHour * CandlesInDay) // 24

    # labelStartCandleHour = startCandleHour+windowLengthHour
    # labelStartCandleIndex = int(labelStartCandleHour * CandlesInDay) // 24

    labelWindowLengthHour = 2
    labelWindowLengthCount = int(labelWindowLengthHour * CandlesInDay) // 24

    trainNps = []
    labelValues = []
    for i in range(0, len(df) - CandlesInDay, CandlesInDay):

        if i % (30 * CandlesInDay) == 0:
            print(i // CandlesInDay)

        currDayDF = df.iloc[
            i + startCandleIndex : (i + startCandleIndex + windowLengthCount)
        ]

        for j in range(len(currDayDF) - dataLengthCount - labelWindowLengthCount - 1):

            startTrain = j
            endTrain = startTrain + dataLengthCount

            startLabel = endTrain + 1
            endLabel = startLabel + labelWindowLengthCount

            trainDF = currDayDF.iloc[startTrain:endTrain]
            labelDF = currDayDF.iloc[startLabel:endLabel]

            lowestIndex = labelDF["<LOW>"].idxmin()

            lowestPriceInLabels = labelDF["<LOW>"][lowestIndex]

            highesIndex = labelDF["<HIGH>"].idxmax()
            highestPriceInLabels = labelDF["<HIGH>"][highesIndex]

            TP = labelDF["<OPEN>"].iloc[0] * takeProfitPoints
            SL = labelDF["<OPEN>"].iloc[0] * stopLoss

            buySignal = 0

            if highestPriceInLabels < TP and lowestPriceInLabels > SL:
                # idea : halfTP new class in labels
                buySignal = 2

            elif highestPriceInLabels < TP and lowestPriceInLabels < SL:
                buySignal = 0
            elif highestPriceInLabels > TP and lowestPriceInLabels > SL:
                buySignal = 1
            else:
                tpIndex = 0
                slIndex = 0
                for k in range(len(labelDF)):
                    if labelDF["<HIGH>"].iloc[k] > TP and labelDF["<LOW>"].iloc[k] > SL:
                        tpIndex = k
                    if labelDF["<HIGH>"].iloc[k] < TP and labelDF["<LOW>"].iloc[k] < SL:
                        slIndex = k
                    if labelDF["<HIGH>"].iloc[k] < TP and labelDF["<LOW>"].iloc[k] > SL:
                        continue
                buySignal = int(tpIndex < slIndex)
                if slIndex == tpIndex:
                    buySignal = 2

            # trainDF =  trainDF /df.abs().max()
            trainDF = (trainDF - df.min()) / (df.max() - df.min())

            # trainDF = trainDF.drop(['<OPEN>','<ATR_24>','<EMA30>','<RSI>','<CCI>','<KELTNER_M>','<KELTNER_L>','<KELTNER_H>','<GREEN>'],axis=1)
            trainDF = trainDF[
                [
                    "<ATR_24_MULT>",
                    "<TICKVOL>",
                    "<KELT_L_IND>",
                    "<KELT_H_IND>",
                    "<BOL_L_IND>",
                    "<BOL_H_IND>",
                ]
            ]

            trainNp = trainDF.to_numpy(dtype=np.float32)
            # trainNp = np.rot90( trainDF.to_numpy())

            labelValues.append(buySignal)
            trainNps.append(trainNp)

    return[trainNps, labelValues]

using System;
using System.IO;
using MLHelpers;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace irisML
{
    public class IrisModel : IMLTrainer
    {
        //Base path of the application
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        //Path to the data file
        private static string _dataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "iris.txt");
        //Path to where the model will be saved
        public static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "irisModel.zip");
        
        //Reference to the MLContext
        private static MLContext _mlContext;
        //Reference to the pipeline of the model
        private static IEstimator<ITransformer> _pipeline;
        //Reference to the model    
        private static ITransformer _model;
        //Training data
        static IDataView _trainData;
        //Testing data
        static IDataView _testData;

        //Constructor for the iris model
        public IrisModel()
        {
            _mlContext = new MLContext(seed: 0);
        }
        //Loads the data
        public void LoadData()
        {
            //Read all the data
            IDataView allData = _mlContext.Data.LoadFromTextFile<IrisData>(path: _dataPath, hasHeader: false, separatorChar: ',');

            //split the data into test and training
            DataOperationsCatalog.TrainTestData splitData = _mlContext.Data.TrainTestSplit(allData, testFraction: 0.3,seed:1);
            _trainData = splitData.TrainSet;
            _testData = splitData.TestSet;  
        }

        //Data pre processing 
        public void BuildPipeline()
        {
            //by default the 'Label' column is considered to be the prediction target
            //Map the 'Species' column to the 'Label' column
            _pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName:nameof(IrisData.Species), outputColumnName:"Label")
            //Set the features to be used 
            .Append(_mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
            //cache the pipeline, this will make downstream processes faster
            .AppendCacheCheckpoint(_mlContext);
        }

        //Build and train the model
        public void BuildAndTrainModel()
        {
            //we can use the Stochastic Dual Coordinate Ascent (SDCA) maximum entropy classification model for our predictions
            _pipeline = _pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
            //Map the output to the PredictedLabel 
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            //Train the model
            _model = _pipeline.Fit(_trainData);
        }

        //Evaluate the performance of the model
        public void EvaluateModel() 
        {
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_model.Transform(_testData));
            Console.WriteLine($"Micro Accuracy: {testMetrics.MicroAccuracy:F2}");
            Console.WriteLine(testMetrics.ConfusionMatrix.GetFormattedConfusionTable());
        }

        //Save the model to file
        public void SaveModelToFile(string pathToFile)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                _mlContext.Model.Save(_model,_trainData.Schema, fs);
            }
        }

        public void Predict(IrisData iris)
        {
            System.Console.WriteLine(_mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(_model).Predict(iris).PredictedLabels);            
        }
    }
}
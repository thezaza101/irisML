using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace irisML
{
    //Data model of iris.txt
    public class IrisPredictor
    {
        ITransformer _model;
        private MLContext _mlContext;
        //Reference to the PredictionEngine for the iris model       
        private PredictionEngine<IrisData, IrisPrediction> _predEngine;

        public IrisPredictor(string pathToModel)
        {
            _mlContext = new MLContext();

            DataViewSchema intpuSchema_variable;
            using (var stream = File.OpenRead(pathToModel))
                _model = _mlContext.Model.Load(stream,out intpuSchema_variable);

            _predEngine = _mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(_model);
        }

        public void Predict(IrisData iris)
        {
            System.Console.WriteLine(_predEngine.Predict(iris).PredictedLabels);            
        }

    }
}





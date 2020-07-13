using System;
using MLHelpers;

namespace irisML
{
    class Program
    {
        static void Main(string[] args)
        {
            IMLTrainer irisModel = new IrisModel();
            irisModel.LoadData();
            irisModel.BuildPipeline();
            irisModel.BuildAndTrainModel();
            irisModel.EvaluateModel();
            irisModel.SaveModelToFile(IrisModel._modelPath); 

            IrisPredictor irisModelPred = new IrisPredictor(IrisModel._modelPath);
            for (int i = 0; i < 5; i++)
            {
                irisModelPred.Predict(AskForIrisData());
            }
        }
        static IrisData AskForIrisData()
        {
            Console.Write("SepalLength: ");
            float sl = float.Parse(Console.ReadLine());
            Console.Write("SepalWidth: ");
            float sw = float.Parse(Console.ReadLine());
            Console.Write("PetalLength: ");
            float pl = float.Parse(Console.ReadLine());
            Console.Write("PetalWidth: ");
            float pw = float.Parse(Console.ReadLine());

            return new IrisData(){
                    SepalLength = sl,
                    SepalWidth = sw,
                    PetalLength = pl,
                    PetalWidth = pw
                };
        }
    }
}

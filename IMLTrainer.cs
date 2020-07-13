namespace MLHelpers
{
    public interface IMLTrainer
    {
        //Loads the data
        void LoadData();

        //Data pre processing 
        void BuildPipeline();

        //Build and train the model
        void BuildAndTrainModel();

        //Evaluate the performance of the model
        void EvaluateModel();

        //Save the model to file
        void SaveModelToFile(string pathToFile);
    }
}
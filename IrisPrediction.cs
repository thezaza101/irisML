using Microsoft.ML.Data;

namespace irisML
{
    // prediction result for iris
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
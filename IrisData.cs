using Microsoft.ML.Data;

namespace irisML
{
    //Data model of iris.txt
    public class IrisData
    {
        [LoadColumn(0),ColumnName("SepalLength")]
        public float SepalLength;

        [LoadColumn(1),ColumnName("SepalWidth")]
        public float SepalWidth;

        [LoadColumn(2),ColumnName("PetalLength")]
        public float PetalLength;

        [LoadColumn(3),ColumnName("PetalWidth")]
        public float PetalWidth;

        [LoadColumn(4),ColumnName("Species")]
        public string Species;
    }
}
namespace IrisClient
{
    public class IrisPrediction
    {
        public float IsSetosa { get; set; }
        public float IsVersicolor { get; set; }
        public float IsVirginica { get; set; }

        public static IrisPrediction FromArray(float[] predictions) => new IrisPrediction
        {
            IsSetosa = predictions[0],
            IsVersicolor = predictions[1],
            IsVirginica = predictions[2]
        };
    }
}

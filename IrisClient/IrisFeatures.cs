namespace IrisClient
{
    public class IrisFeatures
    {
        public float SepalLength { get; set; }
        public float SepalWidth { get; set; }
        public float PetalLength { get; set; }
        public float PetalWidth { get; set; }

        public float[] ToArray() => new float[] { SepalLength, SepalWidth, PetalLength, PetalWidth };

        public static IrisFeatures FromArray(float[] features) => new IrisFeatures
        {
            SepalLength = features[0],
            SepalWidth = features[1],
            PetalLength = features[2],
            PetalWidth = features[3],
        };

        public static IrisFeatures SetosaSample => FromArray(new float[] { 5.1F, 3.3F, 1.7F, 0.5F });
        public static IrisFeatures VersicolorSample => FromArray(new float[] { 5.9F, 3.0F, 4.2F, 1.5F });
        public static IrisFeatures VirginicaSample => FromArray(new float[] { 6.9F, 3.1F, 5.4F, 2.1F });

    }
}

using System.ComponentModel;

namespace IrisClient
{
    public class IrisViewModel : INotifyPropertyChanged
    {
        private IrisFeatures _features = new IrisFeatures();
        private IrisPrediction _prediction = new IrisPrediction();

        public IrisFeatures Features
        {
            get => _features;
            set
            {
                _features = value;
                Notify(nameof(Features));
            }
        }

        public IrisPrediction Prediction
        {
            get => _prediction;
            set
            {
                _prediction = value;
                Notify(nameof(Prediction));
            }
        }

        public void Predict()
        {
            Prediction = Predictor.Predict(Features);
        }

        public event PropertyChangedEventHandler PropertyChanged;
        void Notify(string property) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace IrisClient
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            DataContext = new IrisViewModel();
            InitializeComponent();
        }

        IrisViewModel ViewModel => (IrisViewModel)DataContext;

        private void SampleSetosa(object sender, RoutedEventArgs e) => ViewModel.Features = IrisFeatures.SetosaSample;
        private void SampleVersicolor(object sender, RoutedEventArgs e) => ViewModel.Features = IrisFeatures.VersicolorSample;
        private void SampleVirginica(object sender, RoutedEventArgs e) => ViewModel.Features = IrisFeatures.VirginicaSample;
        private void Predict(object sender, RoutedEventArgs e) => ViewModel.Predict();

    }
}

package neildg.com.eagleeyesr.camera2.util;

import static org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.Arrays;

public class ProcessorTest {

    public static void processImage(String fileName) {
        // Load image using OpenCV's imread
        Mat mat = Imgcodecs.imread(fileName);


        // Check if the image was loaded properly
        if (mat.empty()) {
            System.out.println("Error: Image not loaded! Check file path: " + fileName);
            return;
        }

        // Print pixel values for verification
        System.out.println("Loaded Image Pixel Values:");
        if (mat.empty()) {
            Log.d("TEST", "Mat is empty");
            return;
        }

        int rows = Math.min(mat.rows(), 5);
        int cols = Math.min(mat.cols(), 5);

        Log.d("TEST", "-------------------------------------------------------------------------------");
        Log.d("TEST", "Rows: " + mat.rows() + " Cols: " + mat.cols());
        for (int i = 0; i < rows; i++) {

            StringBuilder rowValues = new StringBuilder("Row " + i + ": ");
            for (int j = 0; j < cols; j++) {
                double[] pixel = mat.get(i, j); // Returns an array for multi-channel images
                rowValues.append(Arrays.toString(pixel)).append(" | ");
            }
            Log.d("TEST", rowValues.toString());
        }
    }
}
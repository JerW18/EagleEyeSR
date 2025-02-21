package neildg.com.eagleeyesr.camera2.util;

import org.opencv.core.Mat;
import android.util.Log;
import java.util.Arrays;

public class Util {
    private static final String TAG = "MatUtils";

    public static void printPixelValues(Mat mat, int maxRows, int maxCols) {
        if (mat.empty()) {
            Log.d(TAG, "Mat is empty");
            return;
        }

        int rows = Math.min(mat.rows(), maxRows);
        int cols = Math.min(mat.cols(), maxCols);

        Log.d(TAG, "-------------------------------------------------------------------------------");
        for (int i = 0; i < rows; i++) {
            StringBuilder rowValues = new StringBuilder("Row " + i + ": ");
            for (int j = 0; j < cols; j++) {
                double[] pixel = mat.get(i, j); // Returns an array for multi-channel images
                rowValues.append(Arrays.toString(pixel)).append(" | ");
            }
            Log.d(TAG, rowValues.toString());
        }
    }

    // Overloaded method with default maxRows and maxCols (5x5)
    public static void printPixelValues(Mat mat) {
        printPixelValues(mat, 5, 5);
    }
}

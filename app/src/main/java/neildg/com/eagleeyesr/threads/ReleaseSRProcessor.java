package neildg.com.eagleeyesr.threads;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.NormalBayesClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Semaphore;

import neildg.com.eagleeyesr.camera2.util.ProcessorTest;
import neildg.com.eagleeyesr.camera2.util.Util;
import neildg.com.eagleeyesr.constants.FilenameConstants;
import neildg.com.eagleeyesr.constants.ParameterConfig;
import neildg.com.eagleeyesr.io.FileImageReader;
import neildg.com.eagleeyesr.io.FileImageWriter;
import neildg.com.eagleeyesr.io.ImageFileAttribute;
import neildg.com.eagleeyesr.io.ImageInputMap;
import neildg.com.eagleeyesr.metrics.TimeMeasure;
import neildg.com.eagleeyesr.metrics.TimeMeasureManager;
import neildg.com.eagleeyesr.model.AttributeHolder;
import neildg.com.eagleeyesr.model.AttributeNames;
import neildg.com.eagleeyesr.model.multiple.SharpnessMeasure;
import neildg.com.eagleeyesr.processing.filters.YangFilter;
import neildg.com.eagleeyesr.processing.imagetools.ImageOperator;
import neildg.com.eagleeyesr.processing.imagetools.MatMemory;
import neildg.com.eagleeyesr.processing.process_observer.IProcessListener;
import neildg.com.eagleeyesr.processing.multiple.alignment.AffineWarpingOperator;
import neildg.com.eagleeyesr.processing.multiple.alignment.FeatureMatchingOperator;
import neildg.com.eagleeyesr.processing.multiple.alignment.LRWarpingOperator;
import neildg.com.eagleeyesr.processing.multiple.alignment.MedianAlignmentOperator;
import neildg.com.eagleeyesr.processing.multiple.alignment.WarpResultEvaluator;
import neildg.com.eagleeyesr.processing.multiple.alignment.WarpingConstants;
import neildg.com.eagleeyesr.processing.multiple.assessment.InputImageEnergyReader;
import neildg.com.eagleeyesr.processing.multiple.enhancement.UnsharpMaskOperator;
import neildg.com.eagleeyesr.processing.multiple.fusion.FusionConstants;
import neildg.com.eagleeyesr.processing.multiple.fusion.MeanFusionOperator;
import neildg.com.eagleeyesr.processing.multiple.refinement.DenoisingOperator;
import neildg.com.eagleeyesr.processing.process_observer.SRProcessManager;
import neildg.com.eagleeyesr.ui.progress_dialog.ProgressDialogHandler;

/**
 * SR processor for release mode
 * Created by NeilDG on 9/10/2016.
 */
public class ReleaseSRProcessor extends Thread{
    private final static String TAG = "ReleaseSRProcessor";

    public ReleaseSRProcessor() {

    }

    @Override
    public void run() {
        ImageInputMap.getPaths();

        TimeMeasure srTimeMeasure = TimeMeasureManager.getInstance().newTimeMeasure(TimeMeasureManager.MEASURE_SR_TIME);
        TimeMeasure edgeDetectionMeasure = TimeMeasureManager.getInstance().newTimeMeasure(TimeMeasureManager.EDGE_DETECTION_TIME);
        TimeMeasure selectionMeasure = TimeMeasureManager.getInstance().newTimeMeasure(TimeMeasureManager.IMAGE_SELECTION_TIME);
        TimeMeasure sharpeningMeasure = TimeMeasureManager.getInstance().newTimeMeasure(TimeMeasureManager.SHARPENING_TIME);
        TimeMeasure denoisingMeasure = TimeMeasureManager.getInstance().newTimeMeasure(TimeMeasureManager.DENOISING_TIME);
        TimeMeasure imageAlignmentMeasure = TimeMeasureManager.getInstance().newTimeMeasure(TimeMeasureManager.IMAGE_ALIGNMENT_TIME);
        TimeMeasure alignmentSelectMeasure = TimeMeasureManager.getInstance().newTimeMeasure(TimeMeasureManager.ALIGNMENT_SELECTION_TIME);
        TimeMeasure fusionMeasuure = TimeMeasureManager.getInstance().newTimeMeasure(TimeMeasureManager.IMAGE_FUSION_TIME);

        srTimeMeasure.timeStart();

        ProgressDialogHandler.getInstance().showProcessDialog("Pre-process", "Analyzing images", 10.0f);

        //initialize classes
        SharpnessMeasure.initialize();
        edgeDetectionMeasure.timeStart();

        for(int i = 0; i < 10; i++) {
            ProcessorTest.processImage(ImageInputMap.getInputImage(i));
        }

        Log.d(TAG, "Reading energy...");
        Mat[] energyInputMatList = new Mat[ImageInputMap.numImages()];
        InputImageEnergyReader[] energyReaders = new InputImageEnergyReader[energyInputMatList.length];
        //load images and use Y channel as input for succeeding operators
        try {
            Semaphore energySem = new Semaphore(energyInputMatList.length);
            for(int i = 0; i < energyReaders.length; i++) {
                energyReaders[i] = new InputImageEnergyReader(energySem, ImageInputMap.getInputImage(i));
                energyReaders[i].startWork();
            }

            energySem.acquire(energyInputMatList.length);
            for(int i = 0; i < energyReaders.length; i++) {
                energyInputMatList[i] = energyReaders[i].getOutputMat();
                Log.d(TAG, "With values after energy reading: " + Core.countNonZero(energyInputMatList[i]));
            }


        } catch(InterruptedException e) {
            e.printStackTrace();
        }


        ProgressDialogHandler.getInstance().showProcessDialog("Pre-process", "Analyzing images", 15.0f);


        for (Mat mat : energyInputMatList) {
            Log.d(TAG, "With values after energy reading, before yang filter: " + Core.countNonZero(mat));
        }
        for (Mat mat : energyInputMatList) {
            Util.printPixelValues(mat);
        }

        Log.d(TAG, "Applying Filter...");
        for (Mat mat : energyInputMatList) {
            Util.printPixelValues(mat);
        }
        //extract features
        YangFilter yangFilter = new YangFilter(energyInputMatList);
        yangFilter.perform();

        edgeDetectionMeasure.timeEnd();

        //release energy input mat list
        MatMemory.releaseAll(energyInputMatList, false);

        for (Mat mat : yangFilter.getEdgeMatList()) {
            Log.d(TAG, "With values after yang filter: " + Core.countNonZero(mat));
        }
        for (Mat mat : yangFilter.getEdgeMatList()) {
            Util.printPixelValues(mat);
        }

        selectionMeasure.timeStart();
        //remeasure sharpness result without the image ground-truth
        Log.d(TAG, "Filteredmatlist length: " + yangFilter.getEdgeMatList().length);
        // IMPORTANT: performSuperResolutionStartsHere
        for (Mat mat : yangFilter.getEdgeMatList()) {
            Util.printPixelValues(mat);
        }

        SharpnessMeasure.SharpnessResult sharpnessResult = SharpnessMeasure.getSharedInstance().measureSharpness(yangFilter.getEdgeMatList());

        //trim the input list from the measured sharpness mean
        Integer[] inputIndices = SharpnessMeasure.getSharedInstance().trimMatList(ImageInputMap.numImages(), sharpnessResult, 0.0);
        Mat[] rgbInputMatList = new Mat[inputIndices.length];

        Log.d(TAG, "Least Index: " + String.valueOf(sharpnessResult.getLeastIndex()));
        Log.d(TAG, "Best Index: " + String.valueOf(sharpnessResult.getBestIndex()));
        Log.d(TAG, "Mean Energy: " + String.valueOf(sharpnessResult.getMean()));
        Log.d(TAG, "Trimmed Indexes: " + Arrays.toString(sharpnessResult.getTrimmedIndexes()));
        Log.d(TAG, "Input Indices: " + Arrays.toString(inputIndices));

        selectionMeasure.timeEnd();

        this.interpolateImage(sharpnessResult.getLeastIndex());
        SRProcessManager.getInstance().initialHRProduced();

        int bestIndex = 0;
        //load RGB inputs
        Mat inputMat;
        sharpeningMeasure.timeStart();
        for(int i = 0; i < inputIndices.length; i++) {
            //rgbInputMatList[i] = FileImageReader.getInstance().imReadFullPath(ImageInputMap.getInputImage(inputIndices[i]));
            //perform unsharp masking
            inputMat = FileImageReader.getInstance().imReadFullPath(ImageInputMap.getInputImage(inputIndices[i]));
            UnsharpMaskOperator unsharpMaskOperator =  new UnsharpMaskOperator(inputMat, inputIndices[i]);
            unsharpMaskOperator.perform();
            rgbInputMatList[i] = unsharpMaskOperator.getResult();
            if(sharpnessResult.getBestIndex() == inputIndices[i]) {
                bestIndex = i;
            }
        }
        sharpeningMeasure.timeEnd();
        Log.d(TAG, Arrays.toString(rgbInputMatList));
        Log.d(TAG, "RGB INPUT LENGTH: "+rgbInputMatList.length+ " Best index: " +bestIndex);

        this.performActualSuperres(rgbInputMatList, inputIndices, bestIndex, false);
        SRProcessManager.getInstance().srProcessCompleted();

        srTimeMeasure.timeEnd();
        Log.i(TAG,"Total processing time is " +TimeMeasureManager.convertDeltaToString(srTimeMeasure.getDeltaDifference()));
        Log.i(TAG, "Edge Detection time: " +TimeMeasureManager.convertDeltaToString(edgeDetectionMeasure.getDeltaDifference()));
        Log.i(TAG, "Image Selection time: " +TimeMeasureManager.convertDeltaToSeconds(selectionMeasure.getDeltaDifference()));
        Log.i(TAG, "Denoising time: " +TimeMeasureManager.convertDeltaToString(denoisingMeasure.getDeltaDifference()));
        Log.i(TAG, "Image Sharpening time: " +TimeMeasureManager.convertDeltaToString(sharpeningMeasure.getDeltaDifference()));
        Log.i(TAG, "Image Alignment time: " +TimeMeasureManager.convertDeltaToString(imageAlignmentMeasure.getDeltaDifference()));
        Log.i(TAG, "Alignment Selection time: " +TimeMeasureManager.convertDeltaToString(alignmentSelectMeasure.getDeltaDifference()));
        Log.i(TAG, "Image Fusion time: " +TimeMeasureManager.convertDeltaToString(fusionMeasuure.getDeltaDifference()));
    }

    public void performActualSuperres(Mat[] rgbInputMatList, Integer[] inputIndices, int bestIndex, boolean debugMode) {
        boolean performDenoising = ParameterConfig.getPrefsBoolean(ParameterConfig.DENOISE_FLAG_KEY, false);

        TimeMeasure denoisingMeasure = TimeMeasureManager.getInstance().getTimeMeasure(TimeMeasureManager.DENOISING_TIME);
        denoisingMeasure.timeStart();
        if(performDenoising) {
            //perform denoising on original input list
            DenoisingOperator denoisingOperator = new DenoisingOperator(rgbInputMatList);
            denoisingOperator.perform();
            MatMemory.releaseAll(rgbInputMatList, false);
            rgbInputMatList = denoisingOperator.getResult();

        }
        else {
            Log.d(TAG, "Denoising will be skipped!");
        }
        denoisingMeasure.timeEnd();


        int srChoice = ParameterConfig.getPrefsInt(ParameterConfig.SR_CHOICE_KEY, FusionConstants.FULL_SR_MODE);
        if(srChoice == FusionConstants.FULL_SR_MODE) {
            this.performFullSRMode(rgbInputMatList, inputIndices, bestIndex, debugMode);
        }
        else {
            MatMemory.releaseAll(rgbInputMatList, false);
            MatMemory.cleanMemory();
            this.performFastSRMode(bestIndex, debugMode);
        }


    }

    public void performFullSRMode(Mat[] rgbInputMatList, Integer[] inputIndices, int bestIndex, boolean debug) {
        //perform feature matching of LR images against the first image as reference mat.
        int warpChoice = ParameterConfig.getPrefsInt(ParameterConfig.WARP_CHOICE_KEY, WarpingConstants.BEST_ALIGNMENT);
        //perform perspective warping and alignment
        Mat[] succeedingMatList =new Mat[rgbInputMatList.length - 1];
        for(int i = 1; i < rgbInputMatList.length; i++) {
            succeedingMatList[i - 1] = rgbInputMatList[i];
        }

        String[] medianResultNames = new String[succeedingMatList.length];
        for(int i = 0; i < medianResultNames.length; i++) {
            medianResultNames[i] = FilenameConstants.MEDIAN_ALIGNMENT_PREFIX + i;
        }

        String[] warpResultnames = new String[succeedingMatList.length];
        for(int i = 0; i < medianResultNames.length; i++) {
            warpResultnames[i] = FilenameConstants.WARP_PREFIX + i;
        }

        TimeMeasure alignmentMeasure = TimeMeasureManager.getInstance().getTimeMeasure(TimeMeasureManager.IMAGE_ALIGNMENT_TIME);
        alignmentMeasure.timeStart();
        if(warpChoice == WarpingConstants.BEST_ALIGNMENT) {
            this.performMedianAlignment(rgbInputMatList, medianResultNames);
            this.performPerspectiveWarping(rgbInputMatList[0], succeedingMatList, succeedingMatList, warpResultnames);
        }
        else if(warpChoice == WarpingConstants.PERSPECTIVE_WARP) {
            //perform perspective warping
            this.performPerspectiveWarping(rgbInputMatList[0], succeedingMatList, succeedingMatList, warpResultnames);
        }
        else {
            this.performMedianAlignment(rgbInputMatList, medianResultNames);
        }
        alignmentMeasure.timeEnd();

        //deallocate some classes
        SharpnessMeasure.destroy();
        MatMemory.cleanMemory();

        int numImages = AttributeHolder.getSharedInstance().getValue(AttributeNames.WARPED_IMAGES_LENGTH_KEY, 0);
        String[] warpedImageNames = new String[numImages];
        String[] medianAlignedNames = new String[numImages];

        for(int i = 0; i < numImages; i++) {
            warpedImageNames[i] = FilenameConstants.WARP_PREFIX +i;
            medianAlignedNames[i] = FilenameConstants.MEDIAN_ALIGNMENT_PREFIX + i;
        }

        TimeMeasure alignSelectMeasure = TimeMeasureManager.getInstance().getTimeMeasure(TimeMeasureManager.ALIGNMENT_SELECTION_TIME);
        alignSelectMeasure.timeStart();
        ProgressDialogHandler.getInstance().showProcessDialog("Processing", "Aligning images", 60.0f);
        String[] alignedImageNames = assessImageWarpResults(inputIndices[0], warpChoice, warpedImageNames, medianAlignedNames, debug);
        alignSelectMeasure.timeEnd();
        MatMemory.cleanMemory();

        TimeMeasure fusionMeasure = TimeMeasureManager.getInstance().getTimeMeasure(TimeMeasureManager.IMAGE_FUSION_TIME);
        fusionMeasure.timeStart();
        ProgressDialogHandler.getInstance().showProcessDialog("Image fusion", "Performing image fusion", 70.0f);
        this.performMeanFusion(inputIndices[0], bestIndex, alignedImageNames, debug);
        fusionMeasure.timeEnd();

        ProgressDialogHandler.getInstance().showProcessDialog("Image fusion", "Performing image fusion", 100.0f);
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        ProgressDialogHandler.getInstance().hideProcessDialog();

        MatMemory.cleanMemory();
    }

    private void performFastSRMode(int bestIndex, boolean debugMode) {
        ProgressDialogHandler.getInstance().showProcessDialog("Image fusion", "Performing image fusion", 60.0f);
        Mat initialMat;
        if(debugMode) {
            initialMat = FileImageReader.getInstance().imReadOpenCV(FilenameConstants.INPUT_PREFIX_STRING + bestIndex, ImageFileAttribute.FileType.JPEG);
        }
        else {
            initialMat = FileImageReader.getInstance().imReadFullPath(ImageInputMap.getInputImage(bestIndex));
        }

        initialMat = ImageOperator.performInterpolation(initialMat, ParameterConfig.getScalingFactor(), Imgproc.INTER_CUBIC);
        FileImageWriter.getInstance().saveMatrixToImage(initialMat, FilenameConstants.HR_SUPERRES, ImageFileAttribute.FileType.JPEG);
        initialMat.release();
        ProgressDialogHandler.getInstance().showProcessDialog("Image fusion", "Performing image fusion", 100.0f);

        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        ProgressDialogHandler.getInstance().hideProcessDialog();

        MatMemory.cleanMemory();
    }

    private void interpolateImage(int index) {
        ProgressDialogHandler.getInstance().showProcessDialog("Pre-process", "Creating initial HR image", 20.0f);
        boolean outputComparisons = ParameterConfig.getPrefsBoolean(ParameterConfig.DEBUGGING_FLAG_KEY, false);

        if(outputComparisons) {
            Mat inputMat = FileImageReader.getInstance().imReadFullPath(ImageInputMap.getInputImage(index));

            Mat outputMat = ImageOperator.performInterpolation(inputMat, ParameterConfig.getScalingFactor(), Imgproc.INTER_LINEAR);
            FileImageWriter.getInstance().saveMatrixToImage(outputMat, FilenameConstants.HR_LINEAR, ImageFileAttribute.FileType.JPEG);
            outputMat.release();

            inputMat.release();
            System.gc();
        }
        else {
            Log.d(TAG, "Debugging mode disabled. Will skip output interpolated images.");
        }
    }

    private static void logMatPixelValues(Mat mat, int rows, int cols) {
        for (int i = 0; i < Math.min(rows, mat.rows()); i++) {
            StringBuilder rowValues = new StringBuilder();
            for (int j = 0; j < Math.min(cols, mat.cols()); j++) {
                double[] pixel = mat.get(i, j);
                rowValues.append(Arrays.toString(pixel)).append(" ");
            }
            Log.d(TAG, "Row " + i + ": " + rowValues.toString());
        }
    }

    public static String[] assessImageWarpResults(int index, int alignmentUsed, String[] warpedImageNames, String[] medianAlignedNames, boolean useLocalDir) {
        if(alignmentUsed == WarpingConstants.BEST_ALIGNMENT) {
            Mat referenceMat;

            if(useLocalDir) {
                referenceMat = FileImageReader.getInstance().imReadOpenCV(FilenameConstants.INPUT_PREFIX_STRING + index, ImageFileAttribute.FileType.JPEG);
            }
            else {
              referenceMat  = FileImageReader.getInstance().imReadFullPath(ImageInputMap.getInputImage(index));
            }

            // Log reference image pixel values
            Log.d(TAG, "Reference image pixel values (first 10x10 block):");
            logMatPixelValues(referenceMat, 10, 10);

            WarpResultEvaluator warpResultEvaluator = new WarpResultEvaluator(referenceMat, warpedImageNames, medianAlignedNames);
            warpResultEvaluator.perform();
            return warpResultEvaluator.getChosenAlignedNames();
        }
        else if(alignmentUsed == WarpingConstants.MEDIAN_ALIGNMENT) {
            return medianAlignedNames;
        }
        else {
            return warpedImageNames;
        }

    }

//    private void performAffineWarping(Mat referenceMat, Mat[] candidateMatList, Mat[] imagesToWarpList) {
//        ProgressDialogHandler.getInstance().showProcessDialog("Processing", "Performing image warping", 30.0f);
//
//        //perform affine warping
//        AffineWarpingOperator warpingOperator = new AffineWarpingOperator(referenceMat, candidateMatList, imagesToWarpList);
//        warpingOperator.perform();
//
//        MatMemory.releaseAll(candidateMatList, false);
//        MatMemory.releaseAll(imagesToWarpList, false);
//        MatMemory.releaseAll(warpingOperator.getWarpedMatList(), true);
//    }

    private void performPerspectiveWarping(Mat referenceMat, Mat[] candidateMatList, Mat[] imagesToWarpList, String[] resultNames) {
        ProgressDialogHandler.getInstance().showProcessDialog("Processing", "Aligning images", 30.0f);
        FeatureMatchingOperator matchingOperator = new FeatureMatchingOperator(referenceMat, candidateMatList);
        matchingOperator.perform();

        ProgressDialogHandler.getInstance().showProcessDialog("Processing", "Aligning images", 40.0f);

        LRWarpingOperator perspectiveWarpOperator = new LRWarpingOperator(matchingOperator.getRefKeypoint(), imagesToWarpList, resultNames, matchingOperator.getdMatchesList(), matchingOperator.getLrKeypointsList());
        perspectiveWarpOperator.perform();

        //release images
        matchingOperator.getRefKeypoint().release();
        MatMemory.releaseAll(matchingOperator.getdMatchesList(), false);
        MatMemory.releaseAll(matchingOperator.getLrKeypointsList(), false);
        MatMemory.releaseAll(candidateMatList, false);
        MatMemory.releaseAll(imagesToWarpList, false);

        Mat[] warpedMatList = perspectiveWarpOperator.getWarpedMatList();
        MatMemory.releaseAll(warpedMatList, false);
    }

    private void performMedianAlignment(Mat[] imagesToAlignList, String[] resultNames) {
        ProgressDialogHandler.getInstance().showProcessDialog("Processing", "Aligning images", 50.0f);
        //perform exposure alignment
        MedianAlignmentOperator medianAlignmentOperator = new MedianAlignmentOperator(imagesToAlignList, resultNames);
        medianAlignmentOperator.perform();

        //MatMemory.releaseAll(imagesToAlignList, true);
    }

    private void performMeanFusion(int index, int bestIndex, String[] alignedImageNames, boolean debugMode) {

        if(alignedImageNames.length == 1) {
            Log.d(TAG, "Best index selected for image HR: " +bestIndex);
            Mat resultMat;
            if(debugMode) {
                resultMat = FileImageReader.getInstance().imReadOpenCV(FilenameConstants.INPUT_PREFIX_STRING + bestIndex, ImageFileAttribute.FileType.JPEG);
            }
            else {
                resultMat = FileImageReader.getInstance().imReadFullPath(ImageInputMap.getInputImage(bestIndex));
            }
            //no need to perform image fusion, just use the best image.
            resultMat = ImageOperator.performInterpolation(resultMat, ParameterConfig.getScalingFactor(), Imgproc.INTER_CUBIC);
            FileImageWriter.getInstance().saveMatrixToImage(resultMat, FilenameConstants.HR_SUPERRES, ImageFileAttribute.FileType.JPEG);

            resultMat.release();
        }
        else {
            ArrayList<String> imagePathList = new ArrayList<>();
            //add initial input HR image
            Mat inputMat;
            if(debugMode) {
                inputMat = FileImageReader.getInstance().imReadOpenCV(FilenameConstants.INPUT_PREFIX_STRING + index, ImageFileAttribute.FileType.JPEG);
            }
            else {
                inputMat = FileImageReader.getInstance().imReadFullPath(ImageInputMap.getInputImage(index));
            }

            for(int i = 0; i < alignedImageNames.length; i++) {
                imagePathList.add(alignedImageNames[i]);
            }

            MeanFusionOperator fusionOperator = new MeanFusionOperator(inputMat, imagePathList.toArray(new String[imagePathList.size()]));
            fusionOperator.perform();
            FileImageWriter.getInstance().saveMatrixToImage(fusionOperator.getResult(), FilenameConstants.HR_SUPERRES, ImageFileAttribute.FileType.JPEG);
            FileImageWriter.getInstance().saveHRResultToUserDir(fusionOperator.getResult(), ImageFileAttribute.FileType.JPEG);

            fusionOperator.getResult().release();
        }




    }
}

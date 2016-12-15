package neildg.com.megatronsr.threads;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;

import neildg.com.megatronsr.constants.FilenameConstants;
import neildg.com.megatronsr.constants.ParameterConfig;
import neildg.com.megatronsr.io.BitmapURIRepository;
import neildg.com.megatronsr.io.ImageFileAttribute;
import neildg.com.megatronsr.io.FileImageReader;
import neildg.com.megatronsr.io.FileImageWriter;
import neildg.com.megatronsr.model.AttributeHolder;
import neildg.com.megatronsr.model.AttributeNames;
import neildg.com.megatronsr.model.multiple.SharpnessMeasure;
import neildg.com.megatronsr.processing.filters.YangFilter;
import neildg.com.megatronsr.processing.imagetools.ColorSpaceOperator;
import neildg.com.megatronsr.processing.imagetools.ImageOperator;
import neildg.com.megatronsr.processing.imagetools.MatMemory;
import neildg.com.megatronsr.processing.listeners.IProcessListener;
import neildg.com.megatronsr.processing.multiple.fusion.OptimizedBaseFusionOperator;
import neildg.com.megatronsr.processing.multiple.refinement.DenoisingOperator;
import neildg.com.megatronsr.processing.multiple.resizing.TransferToDirOperator;
import neildg.com.megatronsr.processing.multiple.warping.AffineWarpingOperator;
import neildg.com.megatronsr.processing.multiple.warping.FeatureMatchingOperator;
import neildg.com.megatronsr.processing.multiple.warping.LRWarpingOperator;
import neildg.com.megatronsr.processing.multiple.warping.WarpResultEvaluator;
import neildg.com.megatronsr.processing.multiple.warping.WarpingConstants;
import neildg.com.megatronsr.ui.ProgressDialogHandler;

/**
 * SR processor for release mode
 * Created by NeilDG on 9/10/2016.
 */
public class ReleaseSRProcessor extends Thread{
    private final static String TAG = "ReleaseSRProcessor";

    private IProcessListener processListener;
    public ReleaseSRProcessor(IProcessListener processListener) {
        this.processListener = processListener;
    }

    @Override
    public void run() {

        ProgressDialogHandler.getInstance().showProcessDialog("Pre-process", "Creating backup copy for processing.", 0.0f);

        TransferToDirOperator transferToDirOperator = new TransferToDirOperator(BitmapURIRepository.getInstance().getNumImagesSelected());
        transferToDirOperator.perform();

        ProgressDialogHandler.getInstance().showProcessDialog("Pre-process", "Interpolating images and extracting energy channel", 10.0f);
        this.interpolateFirstImage();

        //initialize classes
        SharpnessMeasure.initialize();

        //load images and use Y channel as input for succeeding operators
        Mat[] energyInputMatList = new Mat[BitmapURIRepository.getInstance().getNumImagesSelected()];
        Mat inputMat = null;

        for(int i = 0; i < energyInputMatList.length; i++) {
            inputMat = FileImageReader.getInstance().imReadOpenCV(FilenameConstants.INPUT_PREFIX_STRING + (i), ImageFileAttribute.FileType.JPEG);
            inputMat = ImageOperator.downsample(inputMat, 0.125f); //downsample

            FileImageWriter.getInstance().saveMatrixToImage(inputMat, "downsample_"+i, ImageFileAttribute.FileType.JPEG);

            Mat[] yuvMat = ColorSpaceOperator.convertRGBToYUV(inputMat);
            energyInputMatList[i] = yuvMat[ColorSpaceOperator.Y_CHANNEL];

            inputMat.release();

        }

        ProgressDialogHandler.getInstance().showProcessDialog("Processing", "Assessing sharpness measure of images", 15.0f);

        //extract features
        YangFilter yangFilter = new YangFilter(energyInputMatList);
        yangFilter.perform();

        //release energy input mat list
        MatMemory.releaseAll(energyInputMatList, false);

        //remeasure sharpness result without the image ground-truth
        SharpnessMeasure.SharpnessResult sharpnessResult = SharpnessMeasure.getSharedInstance().measureSharpness(yangFilter.getEdgeMatList());

        //trim the input list from the measured sharpness mean
        Integer[] inputIndices = SharpnessMeasure.getSharedInstance().trimMatList(BitmapURIRepository.getInstance().getNumImagesSelected(), sharpnessResult, 0.0);
        Mat[] rgbInputMatList = new Mat[inputIndices.length];

        //load RGB inputs
        for(int i = 0; i < inputIndices.length; i++) {
            rgbInputMatList[i] = FileImageReader.getInstance().imReadOpenCV(FilenameConstants.INPUT_PREFIX_STRING + (inputIndices[i]), ImageFileAttribute.FileType.JPEG);
        }

        Log.d(TAG, "RGB INPUT LENGTH: "+rgbInputMatList.length);

        boolean performDenoising = ParameterConfig.getPrefsBoolean(ParameterConfig.DENOISE_FLAG_KEY, false);

        if(performDenoising) {
            ProgressDialogHandler.getInstance().showProcessDialog("Denoising", "Performing denoising", 20.0f);

            //perform denoising on original input list
            DenoisingOperator denoisingOperator = new DenoisingOperator(rgbInputMatList);
            denoisingOperator.perform();
            MatMemory.releaseAll(rgbInputMatList, false);
            rgbInputMatList = denoisingOperator.getResult();

        }
        else {
            Log.d(TAG, "Denoising will be skipped!");
        }

        //load yang edges for feature matching
        Mat[] candidateMatList = new Mat[inputIndices.length - 1];
        Mat referenceMat = FileImageReader.getInstance().imReadOpenCV(FilenameConstants.EDGE_DIRECTORY_PREFIX + "/" + FilenameConstants.IMAGE_EDGE_PREFIX + 0,
                ImageFileAttribute.FileType.JPEG);

        for(int i = 1; i < inputIndices.length; i++) {
            candidateMatList[i - 1] = FileImageReader.getInstance().imReadOpenCV(FilenameConstants.EDGE_DIRECTORY_PREFIX + "/" + FilenameConstants.IMAGE_EDGE_PREFIX + (inputIndices[i]),
                    ImageFileAttribute.FileType.JPEG);
        }

        Log.d(TAG, "CANDIDATE MAT INPUT LENGTH: "+candidateMatList.length);

        //perform feature matching of LR images against the first image as reference mat.
        Mat[] succeedingMatList =new Mat[rgbInputMatList.length - 1];
        for(int i = 1; i < rgbInputMatList.length; i++) {
            succeedingMatList[i - 1] = rgbInputMatList[i];
        }


        int warpChoice = ParameterConfig.getPrefsInt(ParameterConfig.WARP_CHOICE_KEY, WarpingConstants.AFFINE_WARP);

        if(warpChoice == WarpingConstants.PERSPECTIVE_WARP) {
            //perform perspective warping
            this.performPerspectiveWarping(rgbInputMatList, rgbInputMatList[0], succeedingMatList);
        }
        else {
            //perform affine warping
            this.performAffineWarping(referenceMat, candidateMatList, succeedingMatList);
        }

        //deallocate some classes
        SharpnessMeasure.destroy();

        //ProgressDialogHandler.getInstance().showProcessDialog("Processing", "Refining image warping results", 70.0f);
        //this.assessImageWarpResults();

        ProgressDialogHandler.getInstance().showProcessDialog("Mean fusion", "Performing image fusion", 80.0f);
        this.performMeanFusion();

        ProgressDialogHandler.getInstance().showProcessDialog("Mean fusion", "Performing image fusion", 100.0f);
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        ProgressDialogHandler.getInstance().hideProcessDialog();

        System.gc();

        this.processListener.onProcessCompleted();
    }

    private void interpolateFirstImage() {
        boolean outputComparisons = ParameterConfig.getPrefsBoolean(ParameterConfig.DEBUGGING_FLAG_KEY, false);

        if(outputComparisons) {
            Mat inputMat = FileImageReader.getInstance().imReadOpenCV(FilenameConstants.INPUT_PREFIX_STRING + 0, ImageFileAttribute.FileType.JPEG);

            Mat outputMat = ImageOperator.performInterpolation(inputMat, ParameterConfig.getScalingFactor(), Imgproc.INTER_NEAREST);
            FileImageWriter.getInstance().saveMatrixToImage(outputMat, FilenameConstants.HR_NEAREST, ImageFileAttribute.FileType.JPEG);
            outputMat.release();

        /*outputMat = ImageOperator.performInterpolation(inputMat, ParameterConfig.getScalingFactor(), Imgproc.INTER_LINEAR);
        FileImageWriter.getInstance().saveMatrixToImage(outputMat, FilenameConstants.HR_LINEAR, ImageFileAttribute.FileType.JPEG);
        outputMat.release();*/

            outputMat = ImageOperator.performInterpolation(inputMat, ParameterConfig.getScalingFactor(), Imgproc.INTER_CUBIC);
            FileImageWriter.getInstance().saveMatrixToImage(outputMat, FilenameConstants.HR_CUBIC, ImageFileAttribute.FileType.JPEG);
            outputMat.release();

            inputMat.release();
            System.gc();
        }
        else {
            Log.d(TAG, "Debugging mode disabled. Will skip output interpolated images.");
        }
    }

    private void assessImageWarpResults() {

        int numImages = AttributeHolder.getSharedInstance().getValue(AttributeNames.AFFINE_WARPED_IMAGES_LENGTH_KEY, 0);
        String[] warpedImageNames = new String[numImages];

        for(int i = 0; i < numImages; i++) {
            warpedImageNames[i] = FilenameConstants.WARP_PREFIX +i;
        }

        WarpResultEvaluator warpResultEvaluator = new WarpResultEvaluator(FilenameConstants.INPUT_PREFIX_STRING + 0, warpedImageNames);
        warpResultEvaluator.perform();
    }

    private void performAffineWarping(Mat referenceMat, Mat[] candidateMatList, Mat[] imagesToWarpList) {
        ProgressDialogHandler.getInstance().showProcessDialog("Processing", "Performing image warping", 30.0f);

        //perform affine warping
        AffineWarpingOperator warpingOperator = new AffineWarpingOperator(referenceMat, candidateMatList, imagesToWarpList);
        warpingOperator.perform();

        MatMemory.releaseAll(candidateMatList, false);
        MatMemory.releaseAll(imagesToWarpList, false);
        MatMemory.releaseAll(warpingOperator.getWarpedMatList(), true);
    }

    private void performPerspectiveWarping(Mat[] rgbInputMatList, Mat referenceMat, Mat[] succeedingMatList) {
        ProgressDialogHandler.getInstance().showProcessDialog("Processing", "Performing feature matching against first image", 30.0f);
        FeatureMatchingOperator matchingOperator = new FeatureMatchingOperator(referenceMat, succeedingMatList);
        matchingOperator.perform();

        ProgressDialogHandler.getInstance().showProcessDialog("Processing", "Performing image warping", 60.0f);

        LRWarpingOperator perspectiveWarpOperator = new LRWarpingOperator(matchingOperator.getRefKeypoint(), succeedingMatList, matchingOperator.getdMatchesList(), matchingOperator.getLrKeypointsList());
        perspectiveWarpOperator.perform();

        //release images
        matchingOperator.getRefKeypoint().release();
        MatMemory.releaseAll(matchingOperator.getdMatchesList(), false);
        MatMemory.releaseAll(matchingOperator.getLrKeypointsList(), false);
        MatMemory.releaseAll(succeedingMatList, false);
        MatMemory.releaseAll(rgbInputMatList, false);

        Mat[] warpedMatList = perspectiveWarpOperator.getWarpedMatList();
        MatMemory.releaseAll(warpedMatList, true);
    }

    private void performMeanFusion() {
        int numImages = AttributeHolder.getSharedInstance().getValue(AttributeNames.AFFINE_WARPED_IMAGES_LENGTH_KEY, 0);
        ArrayList<String> imagePathList = new ArrayList<>();

        //add initial input HR image
        imagePathList.add(FilenameConstants.INPUT_PREFIX_STRING + 0);
         for(int i = 0; i < numImages; i++) {
        imagePathList.add(FilenameConstants.WARP_PREFIX +i);
        }

        OptimizedBaseFusionOperator fusionOperator = new OptimizedBaseFusionOperator(imagePathList.toArray(new String[imagePathList.size()]));
        fusionOperator.perform();
        FileImageWriter.getInstance().saveMatrixToImage(fusionOperator.getResult(), FilenameConstants.HR_SUPERRES, ImageFileAttribute.FileType.JPEG);

    }
}

package neildg.com.megatronsr.processing.single;

import android.util.Log;

import neildg.com.megatronsr.model.AttributeHolder;
import neildg.com.megatronsr.model.AttributeNames;
import neildg.com.megatronsr.model.single.ImagePatch;
import neildg.com.megatronsr.model.single.ImagePatchPool;
import neildg.com.megatronsr.processing.IOperator;
import neildg.com.megatronsr.ui.ProgressDialogHandler;

/**
 * Created by NeilDG on 5/9/2016.
 */
public class PatchSimilaritySearch implements IOperator {
    private final static String TAG = "PatchSimilaritySearch";


    public PatchSimilaritySearch() {
        ImagePatchPool.initialize();
    }

    @Override
    public void perform() {
        int pyramidDepth = (int) AttributeHolder.getSharedInstance().getValue(AttributeNames.MAX_PYRAMID_DEPTH_KEY, 0);

        //TODO: testing only
        //PATCH_DIR + this.index, PATCH_PREFIX+col+"_"+row
        String imagePath = PatchExtractCommander.PATCH_DIR+0 + "/" + "patch_0_0";

        ProgressDialogHandler.getInstance().showDialog("Test loading patches", "");

        for(int i = 0; i < 500; i++) {
            ImagePatchPool.getInstance().loadPatch(0, 0, 0, ""+i, imagePath);
            Log.d(TAG, "Loaded patches: " +ImagePatchPool.getInstance().getLoadedPatches());
        }

        ProgressDialogHandler.getInstance().hideDialog();

    }
}

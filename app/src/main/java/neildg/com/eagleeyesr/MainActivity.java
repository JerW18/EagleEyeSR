package neildg.com.eagleeyesr;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.Toast;

import com.darsh.multipleimageselect.activities.AlbumSelectActivity;
import com.darsh.multipleimageselect.helpers.Constants;
import com.darsh.multipleimageselect.models.Image;

import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.util.ArrayList;

import neildg.com.eagleeyesr.constants.ParameterConfig;
import neildg.com.eagleeyesr.io.DirectoryStorage;
import neildg.com.eagleeyesr.io.FileImageReader;
import neildg.com.eagleeyesr.io.FileImageWriter;
import neildg.com.eagleeyesr.model.AttributeHolder;
import neildg.com.eagleeyesr.platformtools.core_application.ApplicationCore;
import neildg.com.eagleeyesr.io.BitmapURIRepository;
import neildg.com.eagleeyesr.ui.progress_dialog.ProgressDialogHandler;
import neildg.com.eagleeyesr.ui.views.AboutScreen;
import neildg.com.eagleeyesr.ui.views.InfoScreen;

public class MainActivity extends AppCompatActivity {

    private final static String TAG = "MainActivity";

    private boolean hasCamera = true;

    private int REQUEST_PICTURE_EXTERNAL = 1;
    private final int PERMISSION_WRITE_EXTERNAL_STORAGE = 2;
    private final int PERMISSION_READ_EXTERNAL_STORAGE = 3;

    static {
        System.loadLibrary("opencv_java4"); // Updated to OpenCV 4.x
        System.loadLibrary("opencv_bridge");
    }

    private InfoScreen infoScreen;
    private AboutScreen aboutScreen;

    private boolean writeGranted = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity_layout);

        this.getSupportActionBar().hide();

        ApplicationCore.initialize(this);
        ProgressDialogHandler.initialize(this);
        ParameterConfig.initialize(this);
        AttributeHolder.initialize(this);

        this.infoScreen = new InfoScreen(this.findViewById(R.id.overlay_intro_view));
        this.infoScreen.initialize();

        this.aboutScreen = new AboutScreen(this.findViewById(R.id.overlay_about_view));
        this.aboutScreen.initialize();
        this.aboutScreen.hide();

        this.verifyCamera();
        this.initializeButtons();
        this.requestPermission();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed");
        } else {
            Log.i(TAG, "OpenCV initialized successfully");
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        ProgressDialogHandler.destroy();
        FileImageWriter.destroy();
        FileImageReader.destroy();
        super.onDestroy();
    }

    private void requestPermission() {
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    PERMISSION_WRITE_EXTERNAL_STORAGE);
        } else {
            this.writeGranted = true;
        }

        if (this.writeGranted) {
            DirectoryStorage.getSharedInstance().createDirectory();
            FileImageWriter.initialize(this);
            FileImageReader.initialize(this);

            Button captureImageBtn = (Button) this.findViewById(R.id.capture_btn);
            captureImageBtn.setEnabled(true);

            Button pickImagesBtn = (Button) this.findViewById(R.id.select_image_btn);
            pickImagesBtn.setEnabled(true);

            Button grantPermissionBtn = (Button) this.findViewById(R.id.button_grant_permission);
            grantPermissionBtn.setVisibility(Button.INVISIBLE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case PERMISSION_WRITE_EXTERNAL_STORAGE: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    this.writeGranted = true;
                }
                break;
            }
        }

        if (this.writeGranted) {
            DirectoryStorage.getSharedInstance().createDirectory();
            FileImageWriter.initialize(this);
            FileImageReader.initialize(this);

            Button captureImageBtn = (Button) this.findViewById(R.id.capture_btn);
            captureImageBtn.setEnabled(true);

            Button pickImagesBtn = (Button) this.findViewById(R.id.select_image_btn);
            pickImagesBtn.setEnabled(true);

            Button grantPermissionBtn = (Button) this.findViewById(R.id.button_grant_permission);
            grantPermissionBtn.setVisibility(Button.INVISIBLE);
        } else {
            Toast.makeText(this, "Eagle-Eye needs to read and write temporary images for processing to your storage.", Toast.LENGTH_LONG)
                    .show();

            Button captureImageBtn = (Button) this.findViewById(R.id.capture_btn);
            captureImageBtn.setEnabled(false);

            Button pickImagesBtn = (Button) this.findViewById(R.id.select_image_btn);
            pickImagesBtn.setEnabled(false);

            Button grantPermissionBtn = (Button) this.findViewById(R.id.button_grant_permission);
            grantPermissionBtn.setVisibility(Button.VISIBLE);
        }
    }

    private void verifyCamera() {
        PackageManager packageManager = ApplicationCore.getInstance().getAppContext().getPackageManager();
        if (!packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA)) {
            Toast.makeText(this, "This device does not have a camera.", Toast.LENGTH_SHORT)
                    .show();
            this.hasCamera = false;
        }
    }

    private void initializeButtons() {
        Button captureImageBtn = (Button) this.findViewById(R.id.capture_btn);
        captureImageBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(MainActivity.this, NewCameraActivity.class);
                startActivity(cameraIntent);
            }
        });

        Button grantPermissionBtn = (Button) this.findViewById(R.id.button_grant_permission);
        grantPermissionBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                MainActivity.this.requestPermission();
            }
        });

        Button pickImagesBtn = (Button) this.findViewById(R.id.select_image_btn);
        pickImagesBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.this.startImagePickActivity();
            }
        });

        final ImageButton infoBtn = (ImageButton) this.findViewById(R.id.about_btn);
        infoBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.this.aboutScreen.show();
            }
        });

        ParameterConfig.setScalingFactor(2);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == Constants.REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            ArrayList<Image> images = data.getParcelableArrayListExtra(Constants.INTENT_EXTRA_IMAGES);
            ArrayList<Uri> imageURIList = new ArrayList<>();
            for (int i = 0; i < images.size(); i++) {
                imageURIList.add(Uri.fromFile(new File(images.get(i).path)));
            }

            if (ParameterConfig.getCurrentTechnique() == ParameterConfig.SRTechnique.MULTIPLE && imageURIList.size() >= 3) {
                Log.v("LOG_TAG", "Selected Images " + imageURIList.size());
                BitmapURIRepository.getInstance().setImageURIList(imageURIList);
                this.moveToProcessingActivity();
            } else if (ParameterConfig.getCurrentTechnique() == ParameterConfig.SRTechnique.MULTIPLE && imageURIList.size() < 3) {
                Toast.makeText(this, "You haven't picked enough images. Pick multiple similar images. At least 3.",
                        Toast.LENGTH_LONG).show();
            }
        } else if (requestCode == REQUEST_PICTURE_EXTERNAL) {
            Log.v(TAG, "Moving to select image activity.");
            this.startImagePickActivity();
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    private void moveToProcessingActivity() {
        Intent processingIntent = new Intent(MainActivity.this, ProcessingActivityRelease.class);
        this.startActivity(processingIntent);
    }

    private void startImagePickActivity() {
        Intent intent = new Intent(MainActivity.this, AlbumSelectActivity.class);
        intent.putExtra(Constants.INTENT_EXTRA_LIMIT, 10);
        startActivityForResult(intent, Constants.REQUEST_CODE);
    }
}
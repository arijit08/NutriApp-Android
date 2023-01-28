package com.arijitbhagavatula.tester1;

import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.arijitbhagavatula.tester1.ml.Nutriapp;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    int IMG_HEIGHT = 224, IMG_WIDTH = 224;
    String[] labels = {"chicken_curry","chicken_wings","fried_rice","grilled_salmon","hamburger","ice_cream","pizza","ramen","steak","sushi"};

    Button classify_btn;
    ImageView food_img;
    TextView prediction_text;

    Bitmap bitmap = null;


    ActivityResultLauncher<String> mGetContent = registerForActivityResult(new ActivityResultContracts.GetContent(),
            new ActivityResultCallback<Uri>() {
                @Override
                public void onActivityResult(Uri uri) {
                    // uri is the path of the image
                    Log.d("TEST", "Path of uri = " + uri.getPath());
                    try {
                        bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(),uri);
                        food_img.setImageBitmap(bitmap);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        classify_btn = (Button) findViewById(R.id.classify_button);
        food_img = (ImageView) findViewById(R.id.food_img);
        prediction_text = (TextView) findViewById(R.id.predict_text);
        classify_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(bitmap!=null){
                    try {
                        ImageProcessor imageProcessor =
                                new ImageProcessor.Builder()
                                        .add(new ResizeOp(IMG_HEIGHT,IMG_WIDTH, ResizeOp.ResizeMethod.BILINEAR))
                                        .add(new NormalizeOp(0, 255))
                                        .build();
                        TensorImage tensorImage = new TensorImage(DataType.UINT8);
                        tensorImage.load(bitmap);
                        tensorImage = imageProcessor.process(tensorImage);
                        Nutriapp model = Nutriapp.newInstance(MainActivity.this);

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
                        inputFeature0.loadBuffer(tensorImage.getBuffer());

                        // Runs model inference and gets result.
                        Nutriapp.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                        float[] pred0 = outputFeature0.getFloatArray();
                        float maxval = -Float.MIN_VALUE;
                        int maxi = 0;
                        for(int i=0;i<10;i++){
                            float val = pred0[i];
                            if(val>maxval){
                                maxval = val;
                                maxi = i;
                            }
                        }
                        prediction_text.setText(labels[maxi]);
                        // Releases model resources if no longer used.
                        model.close();
                    } catch (IOException e) {
                        // TODO Handle the exception
                    }
                }
            }
        });

        food_img.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mGetContent.launch("image/*");
            }
        });
    }
}


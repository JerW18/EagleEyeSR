LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#opencv
# Download OpenCV from https://sourceforge.net/projects/opencvlibrary/files/opencv-android/3.3.0/opencv-3.3.0-android-sdk.zip/download
# Replace this with own path
OPENCVROOT:= C:\Users\Kenshin\Desktop\OpenCV-android-sdk
OPENCV_CAMERA_MODULES:=off
OPENCV_INSTALL_MODULES:=on
OPENCV_LIB_TYPE:=SHARED
include ${OPENCVROOT}/sdk/native/jni/OpenCV.mk

LOCAL_SRC_FILES := sr_jni.cpp
LOCAL_LDLIBS += -llog
LOCAL_MODULE := opencv_bridge

include $(BUILD_SHARED_LIBRARY)
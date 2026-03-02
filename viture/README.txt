Here are the instructions for trying the Unity and iOS app. You can refer to the attached videos for its functionality introduction or setup guidance.

Overall dataflow:
Unity App with NVIDIA StreamKit <-> NVIDIA XR Runtime <-> AI Services.
RGB Camera footage and microphone data is collected by NVIDIA StreamKit and sending to NVIDIA XR Runtime where it could be consumed.
NVIDIA XR runtime has messaging mechanism which messages (text, audio) could be sent to the Unity App.

Instructions:
1. VITURE Pro Neckband software OTA update to version 2.0.8.30211 or later version

2. Install and usage of the updated Unity lab capture app
    I. Download the attached LabCapture0214v2.apk. VITURE may release new versions. This version is for reference.
    II. Install it and grant camera and microphone permissions in the system. Please refer the attached video: Unity-lab-capture-app_permission-setup.mp4
    III. Launch the Unity lab capture app once, so that the following folder will be generated: /storage/emulated/0/Android/data/com.viture.xr.labcapture/files/
    IV. NVIDIA XR runtime IP and PORT configuration file:
        Copy destination path: /storage/emulated/0/Android/data/com.viture.xr.labcapture/files/Config/[Put the sop_config.json here]
        File Name: sop_config.json
    V. [Not used in GTC26 demo version] SOP configuration files:
        Copy destination path: /storage/emulated/0/Android/data/com.viture.xr.labcapture/files/LabSOPs/[Put the sop json files here]
        Multiple SOP files are supported (filename is no longer fixed, but the content format should be following sop_for_demo.json), and the iOS side will directly receive the SOP files pushed by the neckband. (edited) 
    VI. Re-launch the Unity lab capture app, now you can find the SOP and start live streaming.

3. Install and usage of the iOS 3rd person view recording app
    I. Find an iPhone/iPad with iOS 18 or above, and install the TestFlight app (version 0.1.0 build 8)
    II. Install the iOS 3rd person view recording app through https://testflight.apple.com/join/E5Cfy3KT

4. Messaging descriptions
    Please check and follow the event states and the message types in the following descriptions, so that the UI panels' contents will be updated accordingly.

    I. Message types
        The system has the following message types: `AI_RESULT`, `AI_ALERT`, `GENERIC`, `SINGLE_STEP_PANEL_CONTENT`, `COMPONENTS_STATUS`
    II. Payloads
        Please make sure to use DOUBLE QUOTES in json.
        Please check the payload_examples folder for reference.

    III. Payload of AI_RESULT message type (for SOP Panel, not included in the GTC demo version).
        i. Event states
            For each step, it has the following states and transitions. `NOT_STARTED`, `STARTED`,`PAUSED`, `COMPLETED`
            "key": "1", means step 1.
{
    "key":"1",
    "value_state":"COMPLETED",
    "message":"The scientist have added reagent 1 into a EP tube."
}

    IV. Payload of AI_ALERT message type. (This is NOT in the current implementation)
{
    "level": "1",
    "message": "Violation on current operation due to policy x."
}

    V. Payload of GENERIC message type (for LLM Panel). Please send the messages one-by-one. For each piece of message, please use the following format:
{
    "message":
        {
            "type": "rich-text",
            "content": "<A string variable contains rich-text format content.>"，
            "source": "<Agent/User>"
        }
}

    VI. Payload of SINGLE_STEP_PANEL_CONTENT type (for Single-step Panel). We use rich-Text format for text (docs: https://docs.unity3d.com/Packages/com.unity.textmeshpro@4.0/manual/RichTextSubSuper.html). And we use base64 encoding for image.
        The messages are displayed in the order of the array row by row.
        Please check the examples in the `payload_examples/SINGLE_STEP_PANEL_CONTENT` folder. You could use this website for rich-text preview.(https://x3rt.github.io/public/ . The preview website doesn't support the line break tag <br>, but our Unity app supports it.)
{
    "messages": [
        {
            "type": "base64-image",
            "content": "<base64 encoded image string>"
        },
        {
            "type": "rich-text",
            "content": "<A string variable contains rich-text format content.>"
        }
    ]
}

        VII. Payload of COMPONENTS_STATUS type (for components indicators).
{
    "Voice_Assistant": "idle/listening",
    "Server_Connection":"active/inactive",
    "Robot_Status": "<any short string, e.g. running/waiting>"
}

5. Please print the aruco_marker_id0_11cm_letter-paper.pdf file as the marker.

6. Please check the details in the guide video VITURE_LabCapture_instructions_with_marker_tracking.mp4.

7. Please check the new demo video of GTC26 version VITURE_LabCapture_for_GTC26.mp4.

8. New features of Neckband LabCapture app (LabCapture0214v2.apk):
    I. New UI design for GTC26
    II. Updated message types and payloads support
    III. Angled ArUco marker tracking

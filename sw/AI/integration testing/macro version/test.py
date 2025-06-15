import uno
import requests
import json


def send_selected_text_to_ai():
    try:
        # Get the current document and view cursor
        desktop = XSCRIPTCONTEXT.getDesktop()
        model = desktop.getCurrentComponent()

        if not hasattr(model, "Text"):
            return

        xDoc = XSCRIPTCONTEXT.getDocument()
        xController = xDoc.getCurrentController()
        view_cursor = xController.getSelection()

        # Get selected text
        selected_text = view_cursor.getString()
        if not selected_text:
            print("No text selected.")
            return

        # Prepare AI request
        url = "http://localhost:5000/api"  # Replace with your actual AI endpoint
        headers = {"Content-Type": "application/json"}
        payload = {"input": selected_text}

        # Send request to AI backend
        response = requests.post(url, data=json.dumps(payload), headers=headers)

        if response.status_code == 200:
            result = response.json().get("output", "")
            if result:
                # Replace selected text with AI output
                view_cursor.setString(result)
                print("Text replaced with AI response.")
            else:
                print("AI returned no output.")
        else:
            print(f"AI request failed: {response.status_code}")
    except Exception as e:
        print("Error during macro execution:", e)


g_exportedScripts = (send_selected_text_to_ai(),)

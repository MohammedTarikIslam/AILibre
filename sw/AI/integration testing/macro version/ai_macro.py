import uno
from com.sun.star.awt import MessageBoxButtons as MSG_BUTTONS
from com.sun.star.awt.MessageBoxType import MESSAGEBOX
import requests
import json

# import "XScriptContext.idl"


def send_selected_text_to_ai():
    try:

        # Get the current document and view cursor
        desktop = XSCRIPTCONTEXT.getDesktop()
        model = desktop.getCurrentComponent()
        xDoc = XSCRIPTCONTEXT.getDocument()
        xController = xDoc.getCurrentController()
        view_cursor = xController.getSelection()

        ctx = XSCRIPTCONTEXT.getComponentContext()
        smgr = ctx.ServiceManager
        toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
        parent = toolkit.getDesktopWindow()
        # box = toolkit.createMessageBox(parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, "Test", "Macro is working!ucotaes")
        # box.execute()

        selection = xController.getSelection()

        if hasattr(selection, "getString"):
            selected_text = selection.getString()
            box = toolkit.createMessageBox(
                parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, "Notice", "if"
            )
            box.execute()
        elif hasattr(selection, "getCount") and selection.getCount() > 0:
            selected_text = selection.getByIndex(0).getString()
            box = toolkit.createMessageBox(
                parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, "Notice", selected_text
            )
            box.execute()
        else:
            selected_text = ""
            box = toolkit.createMessageBox(
                parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, "Notice", "else"
            )
            box.execute()

        if not selected_text:
            box = toolkit.createMessageBox(
                parent,
                MESSAGEBOX,
                MSG_BUTTONS.BUTTONS_OK,
                "Notice",
                "No valid text selected.",
            )
            box.execute()
            return

        # Prepare AI request
        url = "http://localhost:5000/api"
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


g_exportedScripts = (send_selected_text_to_ai,)

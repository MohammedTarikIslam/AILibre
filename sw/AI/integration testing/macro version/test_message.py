import uno
from com.sun.star.awt import MessageBoxButtons as MSG_BUTTONS
from com.sun.star.awt.MessageBoxType import MESSAGEBOX


def simple_message_box():
    ctx = XSCRIPTCONTEXT.getComponentContext()
    smgr = ctx.ServiceManager
    toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
    parent = toolkit.getDesktopWindow()
    box = toolkit.createMessageBox(
        parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, "Test", "Macro is working! 2"
    )
    box.execute()


g_exportedScripts = (simple_message_box,)

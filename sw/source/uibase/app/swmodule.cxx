/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
/*
 * This file is part of the LibreOffice project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file incorporates work covered by the following license notice:
 *
 *   Licensed to the Apache Software Foundation (ASF) under one or more
 *   contributor license agreements. See the NOTICE file distributed
 *   with this work for additional information regarding copyright
 *   ownership. The ASF licenses this file to you under the Apache
 *   License, Version 2.0 (the "License"); you may not use this file
 *   except in compliance with the License. You may obtain a copy of
 *   the License at http://www.apache.org/licenses/LICENSE-2.0 .
 */

#include <config_features.h>

#include <svtools/ehdl.hxx>
#include <svtools/accessibilityoptions.hxx>
#include <unotools/resmgr.hxx>
#include <unotools/useroptions.hxx>
#include <svl/ctloptions.hxx>
#include <svx/ParaSpacingControl.hxx>
#include <svx/pszctrl.hxx>
#include <svx/insctrl.hxx>
#include <svx/selctrl.hxx>
#include <svx/linectrl.hxx>
#include <svx/tbxctl.hxx>
#include <svx/fillctrl.hxx>
#include <svx/formatpaintbrushctrl.hxx>
#include <svx/contdlg.hxx>
#include <svx/fontwork.hxx>
#include <SwSpellDialogChildWindow.hxx>
#include <svx/grafctrl.hxx>
#include <svx/clipboardctl.hxx>
#include <svx/imapdlg.hxx>
#include <svx/srchdlg.hxx>
#include <svx/hyperdlg.hxx>
#include <svx/modctrl.hxx>
#include <com/sun/star/scanner/ScannerManager.hpp>
#include <com/sun/star/linguistic2/LanguageGuessing.hpp>
#include <ooo/vba/XSinkCaller.hpp>
#include <comphelper/lok.hxx>
#include <comphelper/processfactory.hxx>
#include <docsh.hxx>
#include <swmodule.hxx>
#include <cmdid.h>
#include <pview.hxx>
#include <wview.hxx>
#include <wdocsh.hxx>
#include <srcview.hxx>
#include <glshell.hxx>
#include <tabsh.hxx>
#include <listsh.hxx>
#include <grfsh.hxx>
#include <mediash.hxx>
#include <olesh.hxx>
#include <drawsh.hxx>
#include <wformsh.hxx>
#include <drwtxtsh.hxx>
#include <beziersh.hxx>
#include <wtextsh.hxx>
#include <wfrmsh.hxx>
#include <drformsh.hxx>
#include <wgrfsh.hxx>
#include <wolesh.hxx>
#include <wlistsh.hxx>
#include <wtabsh.hxx>
#include <navipi.hxx>
#include <inputwin.hxx>
#include <usrpref.hxx>
#include <uinums.hxx>
#include <prtopt.hxx>
#include <bookctrl.hxx>
#include <tmplctrl.hxx>
#include <viewlayoutctrl.hxx>
#include <svx/zoomsliderctrl.hxx>
#include <zoomctrl.hxx>
#include <wordcountctrl.hxx>
#include <AccessibilityStatusBarControl.hxx>
#include <workctrl.hxx>
#include <fldwrap.hxx>
#include <redlndlg.hxx>
#include <syncbtn.hxx>
#include <modcfg.hxx>
#include <fontcfg.hxx>
#include <sfx2/sidebar/SidebarChildWindow.hxx>
#include <sfx2/devtools/DevelopmentToolChildWindow.hxx>
#include <swatrset.hxx>
#include <idxmrk.hxx>
#include <wordcountdialog.hxx>
#include <dlelstnr.hxx>
#include <barcfg.hxx>
#include <svx/rubydialog.hxx>
#include <svtools/colorcfg.hxx>

#include <comphelper/configuration.hxx>
#include <unotools/moduleoptions.hxx>

#include <avmedia/mediaplayer.hxx>
#include <avmedia/mediatoolbox.hxx>

#include <annotsh.hxx>
#include <navsh.hxx>

#include <app.hrc>
#include <error.hrc>
#include <strings.hrc>
#include <bitmaps.hlst>
#include <svx/xmlsecctrl.hxx>
bool g_bNoInterrupt = false;

#include <sfx2/app.hxx>

#include <svx/svxerr.hxx>

#include "swdllimpl.hxx"
#include <dbconfig.hxx>
#include <navicfg.hxx>

#include <curl/curl.h>
#include <vcl/weld.hxx>
#include <sfx2/objsh.hxx>
#include <docsh.hxx>
#include <wrtsh.hxx>
#include <rtl/ustring.hxx>
#include <rtl/ustrbuf.hxx>
#include <string>
#include <sstream>
#include <vcl/svapp.hxx>
#include <vcl/dialoghelper.hxx>

using namespace com::sun::star;
using namespace ::com::sun::star::uno;

SwModule::SwModule(SfxObjectFactory* pWebFact, SfxObjectFactory* pFact,
                   SfxObjectFactory* pGlobalFact)
    : SfxModule("sw"_ostr, { pWebFact, pFact, pGlobalFact })
    , m_pView(nullptr)
    , m_eCTLTextNumerals(SvtCTLOptions::GetCTLTextNumerals())
    , m_bAuthorInitialised(false)
    , m_bEmbeddedLoadSave(false)
    , m_pDragDrop(nullptr)
    , m_pXSelection(nullptr)
{
    SetName(u"StarWriter"_ustr);
    SvxErrorHandler::ensure();
    m_pErrorHandler.reset(
        new SfxErrorHandler(RID_SW_ERRHDL, ErrCodeArea::Sw, ErrCodeArea::Sw, GetResLocale()));

    m_pModuleConfig.reset(new SwModuleOptions);

    // We need them anyways
    m_pToolbarConfig.reset(new SwToolbarConfigItem(false));
    m_pWebToolbarConfig.reset(new SwToolbarConfigItem(true));

    m_pStdFontConfig.reset(new SwStdFontConfig);

    {
        SolarMutexGuard g;
        StartListening(*SfxGetpApp());
    }

    if (!comphelper::IsFuzzing())
    {
        // init color configuration
        // member <pColorConfig> is created and the color configuration is applied
        // at the view options.
        GetColorConfig();
        m_xLinguServiceEventListener = new SwLinguServiceEventListener;
    }
}

OUString SwResId(TranslateId aId) { return Translate::get(aId, SwModule::get()->GetResLocale()); }

OUString SwResId(TranslateNId aContextSingularPlural, int nCardinality)
{
    return Translate::nget(aContextSingularPlural, nCardinality, SwModule::get()->GetResLocale());
}

uno::Reference<scanner::XScannerManager2> const& SwModule::GetScannerManager()
{
    static bool bTestScannerManager = true;
    if (bTestScannerManager && !m_xScannerManager.is())
    {
        try
        {
            m_xScannerManager
                = scanner::ScannerManager::create(comphelper::getProcessComponentContext());
        }
        catch (...)
        {
        }
        bTestScannerManager = false;
    }
    return m_xScannerManager;
}

uno::Reference<linguistic2::XLanguageGuessing> const& SwModule::GetLanguageGuesser()
{
    if (!m_xLanguageGuesser.is())
    {
        m_xLanguageGuesser
            = linguistic2::LanguageGuessing::create(comphelper::getProcessComponentContext());
    }
    return m_xLanguageGuesser;
}

SwModule::~SwModule()
{
    css::uno::Sequence<css::uno::Any> aArgs;
    CallAutomationApplicationEventSinks(u"Quit"_ustr, aArgs);
    m_pErrorHandler.reset();
    EndListening(*SfxGetpApp());
}

void SwDLL::RegisterFactories()
{
    // These Id's must not be changed. Through these Id's the View (resume Documentview)
    // is created by Sfx.
    SvtModuleOptions aOptions;
    if (comphelper::IsFuzzing() || aOptions.IsWriterInstalled())
        SwView::RegisterFactory(SFX_INTERFACE_SFXDOCSH);

#if HAVE_FEATURE_DESKTOP
    SwWebView::RegisterFactory(SFX_INTERFACE_SFXMODULE);

    if (comphelper::IsFuzzing() || aOptions.IsWriterInstalled())
    {
        SwSrcView::RegisterFactory(SfxInterfaceId(6));
        SwPagePreview::RegisterFactory(SfxInterfaceId(7));
    }
#endif
}

void SwDLL::RegisterInterfaces()
{
    SwModule* pMod = SwModule::get();
    SwModule::RegisterInterface(pMod);
    SwDocShell::RegisterInterface(pMod);
    SwWebDocShell::RegisterInterface(pMod);
    SwGlosDocShell::RegisterInterface(pMod);
    SwWebGlosDocShell::RegisterInterface(pMod);
    SwView::RegisterInterface(pMod);
    SwWebView::RegisterInterface(pMod);
    SwPagePreview::RegisterInterface(pMod);
    SwSrcView::RegisterInterface(pMod);

    SwBaseShell::RegisterInterface(pMod);
    SwTextShell::RegisterInterface(pMod);
    SwTableShell::RegisterInterface(pMod);
    SwListShell::RegisterInterface(pMod);
    SwFrameShell::RegisterInterface(pMod);
    SwDrawBaseShell::RegisterInterface(pMod);
    SwDrawShell::RegisterInterface(pMod);
    SwDrawFormShell::RegisterInterface(pMod);
    SwDrawTextShell::RegisterInterface(pMod);
    SwBezierShell::RegisterInterface(pMod);
    SwGrfShell::RegisterInterface(pMod);
    SwOleShell::RegisterInterface(pMod);
    SwNavigationShell::RegisterInterface(pMod);
    SwWebTextShell::RegisterInterface(pMod);
    SwWebFrameShell::RegisterInterface(pMod);
    SwWebGrfShell::RegisterInterface(pMod);
    SwWebListShell::RegisterInterface(pMod);
    SwWebTableShell::RegisterInterface(pMod);
    SwWebDrawFormShell::RegisterInterface(pMod);
    SwWebOleShell::RegisterInterface(pMod);
    SwMediaShell::RegisterInterface(pMod);
    SwAnnotationShell::RegisterInterface(pMod);
}

void SwModule::RegisterInterface(SfxModule* pMod)
{
    SFX_NEW_SLOT_ARG(
        SwModule, SID_SEND_SELECTED_TO_AI, SfxGroupId::Document,
        &SwModule::ExecSendSelectedToAI, SFX_STUB_PTR_EXEC_NONE,
        SFX_STUB_PTR_STATE_NONE, SfxSlotMode::SLOT, 0, SfxSlotFlags::NONE)



    static SfxInterface aSwModSlots(aSlots, SAL_N_ELEMENTS(aSlots), pMod);
    pMod->SetInterface(&aSwModSlots);
}

void SwDLL::RegisterControls()
{
    SwModule* pMod = SwModule::get();

    SvxTbxCtlDraw::RegisterControl(SID_INSERT_DRAW, pMod);
    SvxTbxCtlDraw::RegisterControl(SID_TRACK_CHANGES_BAR, pMod);
    SwTbxAutoTextCtrl::RegisterControl(FN_GLOSSARY_DLG, pMod);
    svx::ParaAboveSpacingControl::RegisterControl(SID_ATTR_PARA_ABOVESPACE, pMod);
    svx::ParaBelowSpacingControl::RegisterControl(SID_ATTR_PARA_BELOWSPACE, pMod);
    svx::ParaLeftSpacingControl::RegisterControl(SID_ATTR_PARA_LEFTSPACE, pMod);
    svx::ParaRightSpacingControl::RegisterControl(SID_ATTR_PARA_RIGHTSPACE, pMod);
    svx::ParaFirstLineSpacingControl::RegisterControl(SID_ATTR_PARA_FIRSTLINESPACE, pMod);

    SvxClipBoardControl::RegisterControl(SID_PASTE, pMod);
    svx::FormatPaintBrushToolBoxControl::RegisterControl(SID_FORMATPAINTBRUSH, pMod);

    SvxFillToolBoxControl::RegisterControl(SID_ATTR_FILL_STYLE, pMod);
    SvxLineWidthToolBoxControl::RegisterControl(SID_ATTR_LINE_WIDTH, pMod);

    SwZoomControl::RegisterControl(SID_ATTR_ZOOM, pMod);
    SwPreviewZoomControl::RegisterControl(FN_PREVIEW_ZOOM, pMod);
    SvxPosSizeStatusBarControl::RegisterControl(0, pMod);
    SvxInsertStatusBarControl::RegisterControl(SID_ATTR_INSERT, pMod);
    SvxSelectionModeControl::RegisterControl(FN_STAT_SELMODE, pMod);
    XmlSecStatusBarControl::RegisterControl(SID_SIGNATURE, pMod);
    SwWordCountStatusBarControl::RegisterControl(FN_STAT_WORDCOUNT, pMod);
    sw::AccessibilityStatusBarControl::RegisterControl(FN_STAT_ACCESSIBILITY_CHECK, pMod);

    SwBookmarkControl::RegisterControl(FN_STAT_PAGE, pMod);
    SwTemplateControl::RegisterControl(FN_STAT_TEMPLATE, pMod);
    SwViewLayoutControl::RegisterControl(SID_ATTR_VIEWLAYOUT, pMod);
    SvxModifyControl::RegisterControl(SID_DOC_MODIFIED, pMod);
    SvxZoomSliderControl::RegisterControl(SID_ATTR_ZOOMSLIDER, pMod);

    SvxIMapDlgChildWindow::RegisterChildWindow(false, pMod);
    SvxSearchDialogWrapper::RegisterChildWindow(false, pMod);
    SvxHlinkDlgWrapper::RegisterChildWindow(false, pMod);
    SvxFontWorkChildWindow::RegisterChildWindow(false, pMod);
    SwFieldDlgWrapper::RegisterChildWindow(false, pMod);
    SwFieldDataOnlyDlgWrapper::RegisterChildWindow(false, pMod);
    SvxContourDlgChildWindow::RegisterChildWindow(false, pMod);
    SwInputChild::RegisterChildWindow(false, pMod, SfxChildWindowFlags::FORCEDOCK);
    SwRedlineAcceptChild::RegisterChildWindow(false, pMod);
    SwSyncChildWin::RegisterChildWindow(true, pMod);
    SwInsertIdxMarkWrapper::RegisterChildWindow(false, pMod);
    SwInsertAuthMarkWrapper::RegisterChildWindow(false, pMod);
    SwWordCountWrapper::RegisterChildWindow(false, pMod);
    SvxRubyChildWindow::RegisterChildWindow(false, pMod);
    SwSpellDialogChildWindow::RegisterChildWindow(false, pMod);
    DevelopmentToolChildWindow::RegisterChildWindow(false, pMod);

    SvxGrafRedToolBoxControl::RegisterControl(SID_ATTR_GRAF_RED, pMod);
    SvxGrafGreenToolBoxControl::RegisterControl(SID_ATTR_GRAF_GREEN, pMod);
    SvxGrafBlueToolBoxControl::RegisterControl(SID_ATTR_GRAF_BLUE, pMod);
    SvxGrafLuminanceToolBoxControl::RegisterControl(SID_ATTR_GRAF_LUMINANCE, pMod);
    SvxGrafContrastToolBoxControl::RegisterControl(SID_ATTR_GRAF_CONTRAST, pMod);
    SvxGrafGammaToolBoxControl::RegisterControl(SID_ATTR_GRAF_GAMMA, pMod);
    SvxGrafTransparenceToolBoxControl::RegisterControl(SID_ATTR_GRAF_TRANSPARENCE, pMod);
    SvxGrafModeToolBoxControl::RegisterControl(SID_ATTR_GRAF_MODE, pMod);

#if HAVE_FEATURE_AVMEDIA
    ::avmedia::MediaToolBoxControl::RegisterControl(SID_AVMEDIA_TOOLBOX, pMod);
    ::avmedia::MediaPlayer::RegisterChildWindow(false, pMod);
#endif

    ::sfx2::sidebar::SidebarChildWindow::RegisterChildWindow(false, pMod);

    SwNavigatorWrapper::RegisterChildWindow(false, pMod, SfxChildWindowFlags::NEVERHIDE);

    SwJumpToSpecificPageControl::RegisterControl(SID_JUMP_TO_SPECIFIC_PAGE, pMod);
}

// Load Module (only dummy for linking of the DLL)
void SwModule::InitAttrPool()
{
    OSL_ENSURE(!m_pAttrPool, "Pool already exists!");
    m_pAttrPool = new SwAttrPool(nullptr);
    SetPool(m_pAttrPool.get());
}

void SwModule::RemoveAttrPool()
{
    SetPool(nullptr);
    m_pAttrPool.clear();
}

SfxStyleFamilies SwModule::CreateStyleFamilies()
{
    SfxStyleFamilies aStyleFamilies;

    aStyleFamilies.emplace_back(SfxStyleFamily::Para, SwResId(STR_PARAGRAPHSTYLEFAMILY),
                                BMP_STYLES_FAMILY_PARA, RID_PARAGRAPHSTYLEFAMILY, GetResLocale());

    aStyleFamilies.emplace_back(SfxStyleFamily::Char, SwResId(STR_CHARACTERSTYLEFAMILY),
                                BMP_STYLES_FAMILY_CHAR, RID_CHARACTERSTYLEFAMILY, GetResLocale());

    aStyleFamilies.emplace_back(SfxStyleFamily::Frame, SwResId(STR_FRAMESTYLEFAMILY),
                                BMP_STYLES_FAMILY_FRAME, RID_FRAMESTYLEFAMILY, GetResLocale());

    aStyleFamilies.emplace_back(SfxStyleFamily::Page, SwResId(STR_PAGESTYLEFAMILY),
                                BMP_STYLES_FAMILY_PAGE, RID_PAGESTYLEFAMILY, GetResLocale());

    aStyleFamilies.emplace_back(SfxStyleFamily::Pseudo, SwResId(STR_LISTSTYLEFAMILY),
                                BMP_STYLES_FAMILY_LIST, RID_LISTSTYLEFAMILY, GetResLocale());

    aStyleFamilies.emplace_back(SfxStyleFamily::Table, SwResId(STR_TABLESTYLEFAMILY),
                                BMP_STYLES_FAMILY_TABLE, RID_TABLESTYLEFAMILY, GetResLocale());

    return aStyleFamilies;
}

void SwModule::RegisterAutomationApplicationEventsCaller(
    css::uno::Reference<ooo::vba::XSinkCaller> const& xCaller)
{
    mxAutomationApplicationEventsCaller = xCaller;
}

void SwModule::CallAutomationApplicationEventSinks(const OUString& Method,
                                                   css::uno::Sequence<css::uno::Any>& Arguments)
{
    if (mxAutomationApplicationEventsCaller.is())
        mxAutomationApplicationEventsCaller->CallSinks(Method, Arguments);
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s)
{
    size_t totalSize = size * nmemb;
    s->append((char*)contents, totalSize);
    return totalSize;
}

void SwModule::ExecSendSelectedToAI()
{
    SfxObjectShell* pShell = SfxObjectShell::Current();
    if (!pShell)
        return;

    SwDocShell* pDocShell = dynamic_cast<SwDocShell*>(pShell);
    if (!pDocShell)
        return;

    SwWrtShell* pWrtShell = pDocShell->GetWrtShell();
    if (!pWrtShell)
        return;
        
    weld::Widget* parent = Application::GetActiveTopWindow()->GetFrameWeld();

    OUString selectedText = pWrtShell->GetSelText();
    if (selectedText.isEmpty())
    {
        std::unique_ptr<weld::MessageDialog> xBox(Application::CreateMessageDialog(
            parent, VclMessageType::Info, VclButtonsType::Ok, "No text selected."));
        xBox->run();
        return;
    }

    
    std::unique_ptr<weld::MessageDialog> box(Application::CreateMessageDialog(
        parent,
        VclMessageType::Question,
        VclButtonsType::YesNo,
        OUString("Click YES for Summary mode [default], or NO for Edit mode.")
    ));
    short result = box->run();

    OString prompt;

    if (result == RET_YES)
    {
        prompt
            = OString(
                  "### Instruction:\n"
                  "Read the text carefully. Provide a concise but detailed summary that includes: "
                  "All important assertion, directive, commitment, emotion and declaration\n"
                  "But if the text is only instructions, provide simpler short instructions in "
                  "bullet point format with all details included\n"
                  "### Text:\n")
              + OUStringToOString(selectedText, RTL_TEXTENCODING_UTF8) + "\n\n### Response:\n";
    }
    else
    {
        prompt = OString("### Instruction:\n"
                         "Read the text carefully and improve clarity, grammar, and style without "
                         "changing the factual content.\n"
                         "Preserve formatting where possible. Correct any spelling mistakes and "
                         "grammatical errors.\n"
                         "Ensure the tense is consistent throughout.\n"
                         "### Text:\n")
                 + OUStringToOString(selectedText, RTL_TEXTENCODING_UTF8) + "\n\n### Response:\n";
    }

    std::string responseStr;
    CURL* curl = curl_easy_init();
    if (curl)
    {
        std::string jsonPayload = std::string("{\"prompt\": \"") + prompt.getStr()
                                  + "\", \"stop\": [\"###\"], \"n_predict\": 128, \"temperature\": "
                                    "0.15, \"top_k\": 35, \"top_p\": 0.95 }";
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8080/completion");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonPayload.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseStr);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);

        if (res != CURLE_OK)
        {
            std::unique_ptr<weld::MessageDialog> xBox(Application::CreateMessageDialog(
                parent,
                VclMessageType::Info,
                VclButtonsType::Ok,
                "Error contacting AI server."
            ));
            xBox->run();
        }
    }

    std::string marker = "\"content\":";
    size_t start = responseStr.find(marker);
    OUString finalResp;

    if (start != std::string::npos)
    {
        start += marker.length();
        size_t end = responseStr.find("}", start);
        std::string content = responseStr.substr(start, end - start);
        size_t quoteStart = content.find("\"");
        size_t quoteEnd = content.rfind("\"");
        if (quoteStart != std::string::npos && quoteEnd != std::string::npos
            && quoteEnd > quoteStart)
            content = content.substr(quoteStart + 1, quoteEnd - quoteStart - 1);

        finalResp = OUString::createFromAscii(content.c_str());
    }
    else
    {
        finalResp = "Could not parse AI response.";
    }

    std::unique_ptr<weld::MessageDialog> xBox(Application::CreateMessageDialog(
        parent,
        VclMessageType::Info,
        VclButtonsType::Ok,
        finalResp
    ));
    xBox->run();
}

/* vim:set shiftwidth=4 softtabstop=4 expandtab: */

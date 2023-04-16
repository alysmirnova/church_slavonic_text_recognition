import os
import sys
import wx
from skew_correction import skew_correction
from segment_and_recognize_text_from_image import segment_and_recognize


def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)


class PhotoCtrl(wx.App):
    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)
        self.imageCtrl = None
        self.frame = wx.Frame(None, title='Распознаватель церковнославянского текста')

        self.panel = wx.Panel(self.frame)
        self.MaxWidth = 440
        self.MaxHeight = 500
        wx.Font.AddPrivateFont(resource_path("TriodionUnicode.ttf"))
        self.f = wx.Font(pointSize=14,
                    family=wx.FONTFAMILY_DEFAULT,
                    style=wx.FONTSTYLE_NORMAL,
                    weight=wx.FONTWEIGHT_NORMAL,
                    underline=False,
                    faceName="Triodion Unicode",
                    encoding=wx.FONTENCODING_DEFAULT)

        self.createWidgets()
        self.frame.Show()

    def createWidgets(self):
        img = wx.Image(self.MaxWidth, self.MaxHeight)
        self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY,
                                         wx.Bitmap(img))
        self.photoTxt = wx.TextCtrl(self.panel, size=(200, -1))
        self.text_from_img = wx.StaticText(self.panel, label="Здесь будет распознанный текст")
        self.text_from_img.SetFont(self.f)

        browseBtn = wx.Button(self.panel, label='Открыть')
        browseBtn.Bind(wx.EVT_BUTTON, self.onBrowse)
        saveBtn = wx.Button(self.panel, label='Сохранить текст')
        saveBtn.Bind(wx.EVT_BUTTON, self.onSave)

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer_pic_text = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_buttons = wx.BoxSizer(wx.HORIZONTAL)

        self.sizer_pic_text.Add(self.imageCtrl, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.sizer_pic_text.Add(wx.StaticLine(self.panel, wx.ID_ANY, style=wx.LI_VERTICAL),
                                0, wx.ALL | wx.EXPAND, 5)
        self.sizer_pic_text.Add(self.text_from_img, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.sizer_buttons.Add(self.photoTxt, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.sizer_buttons.Add(browseBtn, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.sizer_buttons.Add(saveBtn, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.mainSizer.Add(self.sizer_pic_text, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.mainSizer.Add(self.sizer_buttons, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)

        self.panel.SetSizer(self.mainSizer)
        self.mainSizer.Fit(self.frame)
        self.panel.Layout()

    def onBrowse(self, event):
        wildcard = 'JPEG файлы (*.jpg)|*.jpg|' + 'PNG файлы (*.png)|*.png'
        dialog = wx.FileDialog(None, "Открыть изображение",
                               wildcard=wildcard,
                               style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.photoTxt.SetValue(dialog.GetPath())
            self.onView()
            self.text_from_img.SetLabel("Распознается...")
            rotate_image = skew_correction(dialog.GetPath())
            text_from_image = segment_and_recognize(rotate_image)
            self.text_from_img.SetLabel(text_from_image)
        dialog.Destroy()

    def onSave(self, event):
        dialog = wx.DirDialog(None, "Выбор директории", style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            text = self.text_from_img.GetLabel()
            with open(os.path.join(dialog.GetPath(), "text from image.txt"), "w", encoding='utf-8') as file:
                file.write(text)

    def onView(self):
        filepath = self.photoTxt.GetValue()
        img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)
        imgsize = img.GetSize()
        w2h_ratio = imgsize[0] / imgsize[1]
        if w2h_ratio > 1:
            target_w = self.MaxWidth
            target_h = target_w / w2h_ratio
            pos = (0, round((target_w - target_h) / 2))
        else:
            target_h = self.MaxHeight
            target_w = target_h * w2h_ratio
            pos = (round((target_h - target_w) / 2), 0)
        bmp = img.Scale(int(target_w), int(target_h), quality=wx.IMAGE_QUALITY_BOX_AVERAGE
                        ).Resize((self.MaxWidth, self.MaxHeight), pos).ConvertToBitmap()
        self.imageCtrl.SetBitmap(wx.Bitmap(bmp))
        self.panel.Refresh()


if __name__ == '__main__':
    app = PhotoCtrl()
    app.MainLoop()

import struct
import sys
import os
import json
import weakref
import pathlib

from PyQt6.QtCore import QCoreApplication, qDebug, Qt, QPoint, QSize
from PyQt6.QtGui import QColor, QOpenGLContext, QSurfaceFormat, QMatrix4x4, QVector4D, QIcon, QWheelEvent, QMouseEvent, QStandardItemModel, QStandardItem
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTreeView, 
                             QLabel, QMessageBox, QWidget, QSplitter, QGroupBox, 
                             QHeaderView, QAbstractItemView)
from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLDebugLogger, QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture, \
    QOpenGLVersionProfile, QOpenGLVertexArrayObject, QOpenGLVersionFunctionsFactory

from DDS.DDSFile import DDSFile

if "mobase" not in sys.modules:
    import mock_mobase as mobase

# Shaders (same as DDSPreview)
vertexShader2D = """
#version 150

uniform float aspectRatioRatio;
uniform mat4 viewMatrix;

in vec4 position;
in vec2 texCoordIn;

out vec2 texCoord;

void main()
{
    texCoord = texCoordIn;
    gl_Position = viewMatrix * position;
    if (aspectRatioRatio >= 1.0)
        gl_Position.y /= aspectRatioRatio;
    else
        gl_Position.x *= aspectRatioRatio;
}
"""

fragmentShaderFloat = """
#version 150

uniform sampler2D aTexture;
uniform mat4 channelMatrix;
uniform vec4 channelOffset;

in vec2 texCoord;

void main()
{
    gl_FragData[0] = channelMatrix * texture(aTexture, texCoord) + channelOffset;
}
"""

fragmentShaderUInt = """
#version 150

uniform usampler2D aTexture;
uniform mat4 channelMatrix;
uniform vec4 channelOffset;

in vec2 texCoord;

void main()
{
    gl_FragData[0] = channelMatrix * texture(aTexture, texCoord) + channelOffset;
}
"""

fragmentShaderSInt = """
#version 150

uniform isampler2D aTexture;
uniform mat4 channelMatrix;
uniform vec4 channelOffset;

in vec2 texCoord;

void main()
{
    gl_FragData[0] = channelMatrix * texture(aTexture, texCoord) + channelOffset;
}
"""

transparencyVS = """
#version 150

uniform mat4 viewMatrix;

in vec4 position;

void main()
{
    gl_Position = viewMatrix * position;
}
"""

transparencyFS = """
#version 150

uniform vec4 backgroundColour;

void main()
{
    float x = gl_FragCoord.x;
    float y = gl_FragCoord.y;
    x = mod(x, 16.0);
    y = mod(y, 16.0);
    gl_FragData[0] = x < 8.0 ^^ y < 8.0 ? vec4(vec3(191.0/255.0), 1.0) : vec4(1.0);
    gl_FragData[0].rgb = backgroundColour.rgb * backgroundColour.a + gl_FragData[0].rgb * (1.0 - backgroundColour.a);
}
"""

vertices = [
    -1.0, -1.0, 0.5, 1.0, 0.0, 1.0,
    -1.0, 1.0, 0.5, 1.0, 0.0, 0.0,
    1.0, 1.0, 0.5, 1.0, 1.0, 0.0,

    -1.0, -1.0, 0.5, 1.0, 0.0, 1.0,
    1.0, 1.0, 0.5, 1.0, 1.0, 0.0,
    1.0, -1.0, 0.5, 1.0, 1.0, 1.0,
]


class DDSOptions:
    def __init__(self, colour: QColor = QColor(0, 0, 0, 0), channelMatrix: QMatrix4x4 = QMatrix4x4(),
                 channelOffset: QVector4D = QVector4D()):
        self.backgroundColour = None
        self.channelMatrix = None
        self.channelOffset = None
        self.setBackgroundColour(colour)
        self.setChannelMatrix(channelMatrix)
        self.setChannelOffset(channelOffset)

    def setBackgroundColour(self, colour: QColor):
        if isinstance(colour, QColor) and colour.isValid():
            self.backgroundColour = colour
        else:
            raise TypeError(str(colour) + " is not a valid QColor object.")

    def getBackgroundColour(self) -> QColor:
        return self.backgroundColour

    def getChannelMatrix(self) -> QMatrix4x4:
        return self.channelMatrix

    def setChannelMatrix(self, matrix):
        self.channelMatrix = QMatrix4x4(matrix)

    def getChannelOffset(self) -> QVector4D:
        return self.channelOffset

    def setChannelOffset(self, vector):
        self.channelOffset = QVector4D(vector)


glVersionProfile = QOpenGLVersionProfile()
glVersionProfile.setVersion(2, 1)


class DDSWidget(QOpenGLWidget):
    def __init__(self, ddsFile, ddsOptions=DDSOptions(), debugContext=False, parent=None, f=Qt.WindowType(0)):
        super(DDSWidget, self).__init__(parent, f)
        self.ddsFile = ddsFile
        self.ddsOptions = ddsOptions
        self.clean = True
        self.logger = None
        self.program = None
        self.transparecyProgram = None
        self.texture = None
        self.vbo = None
        self.vao = None
        
        # Zoom and pan state
        self.zoom = 1.0
        self.minZoom = 1.0
        self.maxZoom = 10.0
        self.panX = 0.0
        self.panY = 0.0
        self.lastMousePos = None
        self.isPanning = False
        
        # View matrix
        self.viewMatrix = QMatrix4x4()
        
        # Enable mouse tracking for panning
        self.setMouseTracking(True)

        self.destroyed.connect(self.cleanup)

        if debugContext:
            format = QSurfaceFormat()
            format.setOption(QSurfaceFormat.FormatOption.DebugContext)
            self.setFormat(format)
            self.logger = QOpenGLDebugLogger(self)

    def __del__(self):
        pass

    def __dtor__(self):
        pass

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        delta = event.angleDelta().y()
        zoomFactor = 1.1 if delta > 0 else 0.9
        oldZoom = self.zoom
        self.zoom *= zoomFactor
        self.zoom = max(self.minZoom, min(self.maxZoom, self.zoom))
        
        if self.zoom != oldZoom:
            mousePos = event.position()
            widgetCenterX = self.width() / 2.0
            widgetCenterY = self.height() / 2.0
            normX = (mousePos.x() - widgetCenterX) / widgetCenterX
            normY = -(mousePos.y() - widgetCenterY) / widgetCenterY
            actualZoomChange = self.zoom / oldZoom
            self.panX = self.panX * actualZoomChange + normX * (1.0 - actualZoomChange) / self.zoom
            self.panY = self.panY * actualZoomChange + normY * (1.0 - actualZoomChange) / self.zoom
            self._updateViewMatrix()
            self.update()
        
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        """Start panning on middle or right mouse button"""
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self.isPanning = True
            self.lastMousePos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Stop panning"""
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self.isPanning = False
            self.lastMousePos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle panning"""
        if self.isPanning and self.lastMousePos is not None:
            delta = event.pos() - self.lastMousePos
            self.lastMousePos = event.pos()
            self.panX += (delta.x() / self.width()) * 5.0 / self.zoom
            self.panY -= (delta.y() / self.height()) * 5.0 / self.zoom
            self._updateViewMatrix()
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Reset zoom and pan on double click"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.zoom = 1.0
            self.panX = 0.0
            self.panY = 0.0
            self._updateViewMatrix()
            self.update()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def _updateViewMatrix(self):
        """Update the view transformation matrix"""
        self.viewMatrix = QMatrix4x4()
        self.viewMatrix.translate(self.panX, self.panY, 0.0)
        self.viewMatrix.scale(self.zoom, self.zoom, 1.0)

    def initializeGL(self):
        if self.logger:
            self.logger.initialize()
            self.logger.messageLogged.connect(
                lambda message: qDebug(self.tr("OpenGL debug message: {0}").format(message.message())))
            self.logger.startLogging()

        gl = QOpenGLVersionFunctionsFactory.get(glVersionProfile)

        self.clean = False

        fragmentShader = None
        if self.ddsFile.glFormat.samplerType == "F":
            fragmentShader = fragmentShaderFloat
        elif self.ddsFile.glFormat.samplerType == "UI":
            fragmentShader = fragmentShaderUInt
        else:
            fragmentShader = fragmentShaderSInt

        self.program = QOpenGLShaderProgram(self)
        self.program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vertexShader2D)
        self.program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fragmentShader)
        self.program.bindAttributeLocation("position", 0)
        self.program.bindAttributeLocation("texCoordIn", 1)
        self.program.link()

        self.transparecyProgram = QOpenGLShaderProgram(self)
        self.transparecyProgram.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, transparencyVS)
        self.transparecyProgram.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, transparencyFS)
        self.transparecyProgram.bindAttributeLocation("position", 0)
        self.transparecyProgram.link()

        self.vao = QOpenGLVertexArrayObject(self)
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)

        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.vbo.create()
        self.vbo.bind()

        theBytes = struct.pack("%sf" % len(vertices), *vertices)
        self.vbo.allocate(theBytes, len(theBytes))

        gl.glEnableVertexAttribArray(0)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, False, 6 * 4, 0)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, 6 * 4, 4 * 4)

        self.texture = self.ddsFile.asQOpenGLTexture(gl, QOpenGLContext.currentContext())
        
        self._updateViewMatrix()

    def resizeGL(self, w, h):
        aspectRatioTex = self.texture.width() / self.texture.height() if self.texture else 1.0
        aspectRatioWidget = w / h
        ratioRatio = aspectRatioTex / aspectRatioWidget

        self.program.bind()
        self.program.setUniformValue("aspectRatioRatio", ratioRatio)
        self.program.release()

    def paintGL(self):
        gl = QOpenGLVersionFunctionsFactory.get(glVersionProfile)
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)

        self.transparecyProgram.bind()
        self.transparecyProgram.setUniformValue("viewMatrix", self.viewMatrix)
        backgroundColour = self.ddsOptions.getBackgroundColour()
        if backgroundColour and backgroundColour.isValid():
            self.transparecyProgram.setUniformValue("backgroundColour", backgroundColour)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        self.transparecyProgram.release()

        self.program.bind()
        self.program.setUniformValue("viewMatrix", self.viewMatrix)
        if self.texture:
            self.texture.bind()

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.program.setUniformValue("channelMatrix", self.ddsOptions.getChannelMatrix())
        self.program.setUniformValue("channelOffset", self.ddsOptions.getChannelOffset())

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        if self.texture:
            self.texture.release()
        self.program.release()

    def cleanup(self):
        """Properly clean up OpenGL resources"""
        if self.clean:
            return
            
        self.clean = True
        
        try:
            context = self.context()
            if context and context.isValid():
                try:
                    self.makeCurrent()
                except RuntimeError as e:
                    print(f"DEBUG: Failed to make context current: {str(e)}")
                    self.program = None
                    self.transparecyProgram = None
                    self.texture = None
                    self.vbo = None
                    self.vao = None
                    return

                if hasattr(self, 'program') and self.program:
                    self.program.release()
                    self.program = None

                if hasattr(self, 'transparecyProgram') and self.transparecyProgram:
                    self.transparecyProgram.release()
                    self.transparecyProgram = None

                if hasattr(self, 'texture') and self.texture and self.context().isValid():
                    try:
                        self.texture.destroy()
                    except RuntimeError as e:
                        print(f"DEBUG: Failed to destroy texture: {str(e)}")
                    self.texture = None

                if hasattr(self, 'vbo') and self.vbo:
                    self.vbo.destroy()
                    self.vbo = None

                if hasattr(self, 'vao') and self.vao:
                    self.vao.destroy()
                    self.vao = None

                self.doneCurrent()
            else:
                print("DEBUG: OpenGL context is invalid or unavailable, skipping GPU resource cleanup")
                self.program = None
                self.transparecyProgram = None
                self.texture = None
                self.vbo = None
                self.vao = None
        except (RuntimeError, SystemError, AttributeError) as e:
            print(f"DEBUG: Error during cleanup: {str(e)}")

    def tr(self, str):
        return QCoreApplication.translate("DDSWidget", str)


class HiddenFileItem:
    """Represents a hidden file entry"""
    def __init__(self, mod_name, mod_path, original_path, hidden_path):
        self.mod_name = mod_name
        self.mod_path = mod_path
        self.original_path = original_path
        self.hidden_path = hidden_path
        self.exists = os.path.exists(hidden_path)
        self.size = os.path.getsize(hidden_path) if self.exists else 0


class HiddenFilesManagerDialog(QDialog):
    def __init__(self, organizer, parent=None):
        super().__init__(parent)
        self.organizer = organizer
        self.preview_widgets = []
        self.hidden_files = []
        self.options = DDSOptions()
        
        self.setWindowTitle(self.tr("DDS Hidden Files Manager"))
        self.setMinimumSize(1000, 600)
        
        self._initUI()
        self._loadHiddenFiles()
    
    def __del__(self):
        self._cleanupPreviews()
    
    def _cleanupPreviews(self):
        """Clean up all preview widgets"""
        for widget_ref in self.preview_widgets[:]:
            widget = widget_ref() if isinstance(widget_ref, weakref.ref) else widget_ref
            if widget is not None:
                try:
                    if isinstance(widget, DDSWidget) and not widget.clean:
                        widget.cleanup()
                    widget.deleteLater()
                except (RuntimeError, SystemError, AttributeError) as e:
                    print(f"DEBUG: Error cleaning up preview widget: {str(e)}")
                finally:
                    self.preview_widgets.remove(widget_ref)
        self.preview_widgets.clear()
    
    def _initUI(self):
        main_layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel(self.tr(
            "This manager shows all hidden DDS files across all mods.\n"
            "Hidden files have been renamed to *.dds.mohidden and are tracked in dds_actions.json"
        ))
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 10px; background-color: #e3f2fd; border: 1px solid #2196f3; border-radius: 3px;")
        main_layout.addWidget(info_label)
        
        # Splitter for tree and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Tree view
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        # Tree view
        self.tree_view = QTreeView()
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree_view.clicked.connect(self._onItemClicked)
        left_layout.addWidget(self.tree_view)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.restore_btn = QPushButton(self.tr("Restore Selected"))
        self.restore_btn.setEnabled(False)
        self.restore_btn.clicked.connect(self._restoreSelected)
        btn_layout.addWidget(self.restore_btn)
        
        self.restore_all_btn = QPushButton(self.tr("Restore All in Mod"))
        self.restore_all_btn.setEnabled(False)
        self.restore_all_btn.clicked.connect(self._restoreAllInMod)
        btn_layout.addWidget(self.restore_all_btn)
        
        refresh_btn = QPushButton(self.tr("Refresh"))
        refresh_btn.clicked.connect(self._refresh)
        btn_layout.addWidget(refresh_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        left_layout.addLayout(btn_layout)
        
        splitter.addWidget(left_widget)
        
        # Right panel: Preview
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        preview_group = QGroupBox(self.tr("Preview"))
        preview_layout = QVBoxLayout()
        preview_group.setLayout(preview_layout)
        
        self.preview_container = QWidget()
        self.preview_layout = QVBoxLayout()
        self.preview_container.setLayout(self.preview_layout)
        preview_layout.addWidget(self.preview_container)
        
        self.preview_info_label = QLabel(self.tr("Select a file to preview"))
        self.preview_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_info_label.setStyleSheet("color: #666; font-style: italic; padding: 20px;")
        self.preview_layout.addWidget(self.preview_info_label)
        
        right_layout.addWidget(preview_group)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter, 1)
        
        self.setLayout(main_layout)
    
    def _loadHiddenFiles(self):
        """Load all hidden files from all mods"""
        self.hidden_files.clear()
        
        mods_list = self.organizer.modList()
        mod_names = mods_list.allModsByProfilePriority()
        
        for mod_name in mod_names:
            mod = mods_list.getMod(mod_name)
            if not mod:
                continue
            
            mod_path = mod.absolutePath()
            json_path = os.path.join(mod_path, 'dds_actions.json')
            
            if not os.path.exists(json_path):
                continue
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    hidden_list = data.get('hidden', [])
                    
                    # Validate and clean up
                    valid_hidden = []
                    needs_update = False
                    
                    for entry in hidden_list:
                        original = entry.get('original', '')
                        hidden = entry.get('hidden', '')
                        
                        if not hidden:
                            needs_update = True
                            continue
                        
                        if os.path.exists(hidden):
                            hidden_item = HiddenFileItem(mod_name, mod_path, original, hidden)
                            self.hidden_files.append(hidden_item)
                            valid_hidden.append(entry)
                        else:
                            # File doesn't exist, remove from JSON
                            needs_update = True
                    
                    # Update JSON if needed
                    if needs_update:
                        data['hidden'] = valid_hidden
                        with open(json_path, 'w') as f:
                            json.dump(data, f, indent=4)
                        
            except Exception as e:
                print(f"DEBUG: Error loading JSON for mod {mod_name}: {str(e)}")
        
        self._updateTreeView()
    
    def _updateTreeView(self):
        """Update tree view with loaded hidden files"""
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels([self.tr("Mod / File"), self.tr("Size"), self.tr("Status")])
        
        # Group by mod
        mods_dict = {}
        for hidden_file in self.hidden_files:
            if hidden_file.mod_name not in mods_dict:
                mods_dict[hidden_file.mod_name] = []
            mods_dict[hidden_file.mod_name].append(hidden_file)
        
        # Build tree
        for mod_name, files in sorted(mods_dict.items()):
            mod_item = QStandardItem(f"📁 {mod_name}")
            mod_item.setData({"type": "mod", "mod_name": mod_name}, Qt.ItemDataRole.UserRole)
            size_item = QStandardItem(f"{len(files)} file(s)")
            status_item = QStandardItem("")
            
            for hidden_file in files:
                file_name = os.path.basename(hidden_file.hidden_path)
                file_item = QStandardItem(f"  🔒 {file_name}")
                file_item.setData({"type": "file", "hidden_file": hidden_file}, Qt.ItemDataRole.UserRole)
                
                size_str = self._formatSize(hidden_file.size)
                file_size_item = QStandardItem(size_str)
                
                status_str = "✓ Ready" if hidden_file.exists else "✗ Missing"
                file_status_item = QStandardItem(status_str)
                if not hidden_file.exists:
                    file_status_item.setForeground(Qt.GlobalColor.red)
                
                mod_item.appendRow([file_item, file_size_item, file_status_item])
            
            model.appendRow([mod_item, size_item, status_item])
        
        self.tree_view.setModel(model)
        self.tree_view.expandAll()
        
        # Resize columns
        header = self.tree_view.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        # Update status label
        total_files = len(self.hidden_files)
        total_mods = len(mods_dict)
        self.setWindowTitle(self.tr(f"DDS Hidden Files Manager - {total_files} hidden file(s) in {total_mods} mod(s)"))
    
    def _onItemClicked(self, index):
        """Handle tree item click"""
        model = self.tree_view.model()
        item = model.itemFromIndex(index)
        data = item.data(Qt.ItemDataRole.UserRole)
        
        if not data:
            self.restore_btn.setEnabled(False)
            self.restore_all_btn.setEnabled(False)
            self._clearPreview()
            return
        
        if data["type"] == "mod":
            self.restore_btn.setEnabled(False)
            self.restore_all_btn.setEnabled(True)
            self._clearPreview()
            self.preview_info_label.setText(self.tr(f"Mod: {data['mod_name']}\nSelect a file to preview"))
            self.preview_info_label.setVisible(True)
        elif data["type"] == "file":
            hidden_file = data["hidden_file"]
            self.restore_btn.setEnabled(hidden_file.exists)
            self.restore_all_btn.setEnabled(False)
            self._showPreview(hidden_file)
    
    def _showPreview(self, hidden_file):
        """Show preview of hidden file"""
        self._clearPreview()
        
        if not hidden_file.exists:
            error_label = QLabel(self.tr("File not found - cannot preview"))
            error_label.setStyleSheet("color: red; padding: 20px;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_layout.addWidget(error_label)
            return
        
        try:
            dds_file = DDSFile.fromFile(hidden_file.hidden_path)
            dds_file.load()
            
            dds_widget = DDSWidget(dds_file, self.options, False)
            dds_widget.setMinimumHeight(300)
            
            self.preview_widgets.append(weakref.ref(dds_widget))
            
            info_label = QLabel(self.tr(
                f"File: {os.path.basename(hidden_file.hidden_path)}\n"
                f"Size: {self._formatSize(hidden_file.size)}\n"
                f"Mod: {hidden_file.mod_name}\n\n"
                f"💡 Use mouse wheel to zoom, right/middle click + drag to pan"
            ))
            info_label.setStyleSheet("color: #666; padding: 5px; background-color: #f5f5f5;")
            info_label.setWordWrap(True)
            self.preview_layout.addWidget(info_label)
            
            self.preview_layout.addWidget(dds_widget)
            
        except Exception as e:
            error_label = QLabel(self.tr(f"Error loading preview:\n{str(e)}"))
            error_label.setStyleSheet("color: red; padding: 10px;")
            error_label.setWordWrap(True)
            self.preview_layout.addWidget(error_label)
    
    def _clearPreview(self):
        """Clear preview area"""
        while self.preview_layout.count():
            item = self.preview_layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                try:
                    if isinstance(widget, DDSWidget) and not widget.clean:
                        widget.cleanup()
                    widget.deleteLater()
                except (RuntimeError, SystemError, AttributeError) as e:
                    print(f"DEBUG: Error cleaning up preview widget: {str(e)}")
                self.preview_widgets = [w for w in self.preview_widgets 
                                      if (w() if isinstance(w, weakref.ref) else w) != widget]
        
        self.preview_info_label = QLabel(self.tr("Select a file to preview"))
        self.preview_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_info_label.setStyleSheet("color: #666; font-style: italic; padding: 20px;")
        self.preview_layout.addWidget(self.preview_info_label)
    
    def _restoreSelected(self):
        """Restore selected hidden file"""
        index = self.tree_view.currentIndex()
        if not index.isValid():
            return
        
        model = self.tree_view.model()
        item = model.itemFromIndex(index)
        data = item.data(Qt.ItemDataRole.UserRole)
        
        if not data or data["type"] != "file":
            return
        
        hidden_file = data["hidden_file"]
        
        if not hidden_file.exists:
            QMessageBox.warning(
                self,
                self.tr("File Not Found"),
                self.tr(f"Hidden file does not exist:\n{hidden_file.hidden_path}")
            )
            return
        
        answer = QMessageBox.question(
            self,
            self.tr("Confirm Restore"),
            self.tr(f"Restore file in mod '{hidden_file.mod_name}'?\n\n"
                   f"From: {os.path.basename(hidden_file.hidden_path)}\n"
                   f"To: {os.path.basename(hidden_file.original_path)}"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if answer != QMessageBox.StandardButton.Yes:
            return
        
        try:
            self._cleanupPreviews()
            
            # Check if destination exists
            if os.path.exists(hidden_file.original_path):
                overwrite = QMessageBox.question(
                    self,
                    self.tr("File Exists"),
                    self.tr(f"File already exists:\n{hidden_file.original_path}\n\nOverwrite?"),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if overwrite != QMessageBox.StandardButton.Yes:
                    return
                os.remove(hidden_file.original_path)
            
            # Restore file
            os.rename(hidden_file.hidden_path, hidden_file.original_path)
            
            # Update JSON
            self._updateJson(hidden_file.mod_path, 'restore', hidden_file.original_path, hidden_file.hidden_path)
            
            # Notify MO2
            mod = self.organizer.modList().getMod(hidden_file.mod_name)
            if mod:
                self.organizer.modDataChanged(mod)
            
            QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr(f"File restored successfully:\n{hidden_file.original_path}")
            )
            
            self._refresh()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr(f"Failed to restore file:\n{str(e)}")
            )
    
    def _restoreAllInMod(self):
        """Restore all hidden files in selected mod"""
        index = self.tree_view.currentIndex()
        if not index.isValid():
            return
        
        model = self.tree_view.model()
        item = model.itemFromIndex(index)
        data = item.data(Qt.ItemDataRole.UserRole)
        
        if not data or data["type"] != "mod":
            return
        
        mod_name = data["mod_name"]
        
        # Get all files for this mod
        files_to_restore = [f for f in self.hidden_files if f.mod_name == mod_name and f.exists]
        
        if not files_to_restore:
            QMessageBox.information(
                self,
                self.tr("No Files"),
                self.tr(f"No files to restore in mod '{mod_name}'")
            )
            return
        
        answer = QMessageBox.question(
            self,
            self.tr("Confirm Restore All"),
            self.tr(f"Restore all {len(files_to_restore)} hidden file(s) in mod '{mod_name}'?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if answer != QMessageBox.StandardButton.Yes:
            return
        
        self._cleanupPreviews()
        
        success_count = 0
        error_count = 0
        errors = []
        
        for hidden_file in files_to_restore:
            try:
                # Check if destination exists
                if os.path.exists(hidden_file.original_path):
                    os.remove(hidden_file.original_path)
                
                # Restore file
                os.rename(hidden_file.hidden_path, hidden_file.original_path)
                
                # Update JSON
                self._updateJson(hidden_file.mod_path, 'restore', hidden_file.original_path, hidden_file.hidden_path)
                
                success_count += 1
                
            except Exception as e:
                error_count += 1
                errors.append(f"{os.path.basename(hidden_file.hidden_path)}: {str(e)}")
        
        # Notify MO2
        mod = self.organizer.modList().getMod(mod_name)
        if mod:
            self.organizer.modDataChanged(mod)
        
        # Show results
        if error_count == 0:
            QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr(f"All {success_count} file(s) restored successfully!")
            )
        else:
            error_text = "\n".join(errors[:5])
            if len(errors) > 5:
                error_text += f"\n... and {len(errors) - 5} more errors"
            
            QMessageBox.warning(
                self,
                self.tr("Partial Success"),
                self.tr(f"Restored: {success_count}\nFailed: {error_count}\n\nErrors:\n{error_text}")
            )
        
        self._refresh()
    
    def _updateJson(self, mod_path, action, original_path, hidden_path):
        """Update JSON file"""
        json_path = os.path.join(mod_path, 'dds_actions.json')
        data = {'hidden': []}
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except:
                pass
        
        hidden_list = data.get('hidden', [])
        
        if action == 'restore':
            # Remove entry
            hidden_list = [h for h in hidden_list if h.get('hidden', '') != hidden_path]
        
        data['hidden'] = hidden_list
        
        try:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"DEBUG: Error updating JSON: {str(e)}")
    
    def _refresh(self):
        """Refresh the list"""
        self._clearPreview()
        self._loadHiddenFiles()
        self.restore_btn.setEnabled(False)
        self.restore_all_btn.setEnabled(False)
    
    def _formatSize(self, size):
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
    
    def tr(self, str_):
        return QCoreApplication.translate("HiddenFilesManagerDialog", str_)
    
    def closeEvent(self, event):
        """Clean up on close"""
        self._cleanupPreviews()
        super().closeEvent(event)


class DDSHiddenFilesManager(mobase.IPluginTool):
    def __init__(self):
        super().__init__()
        self.__organizer = None
        self.__parent_widget = None
    
    def init(self, organizer):
        self.__organizer = organizer
        return True
    
    def name(self):
        return "DDS Hidden Files Manager"
    
    def localizedName(self):
        return self.tr("DDS Hidden Files Manager")
    
    def author(self):
        return "AnyOldName3"
    
    def description(self):
        return self.tr("Manage and restore hidden DDS files across all mods. "
                      "Shows files that were hidden using the DDS Preview plugin.")
    
    def version(self):
        return mobase.VersionInfo(1, 0, 0, 0)
    
    def settings(self):
        return []
    
    def displayName(self):
        return self.tr("DDS Hidden Files Manager")
    
    def tooltip(self):
        return self.tr("Manage and restore hidden DDS files")
    
    def icon(self):
        return QIcon()
    
    def setParentWidget(self, widget):
        self.__parent_widget = widget
    
    def display(self):
        """Display the manager dialog"""
        if not self.__organizer:
            QMessageBox.critical(
                self.__parent_widget,
                self.tr("Error"),
                self.tr("Organizer not initialized")
            )
            return
        
        dialog = HiddenFilesManagerDialog(self.__organizer, self.__parent_widget)
        dialog.exec()
    
    def tr(self, str_):
        return QCoreApplication.translate("DDSHiddenFilesManager", str_)


def createPlugin():
    return DDSHiddenFilesManager()
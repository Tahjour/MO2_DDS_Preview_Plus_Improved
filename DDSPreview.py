import struct
import sys
import enum
import os
import shutil
import time
import weakref
import json

from PyQt6.QtCore import QCoreApplication, qDebug, Qt, QPoint, QSize, QSettings, pyqtSignal
from PyQt6.QtGui import QColor, QOpenGLContext, QSurfaceFormat, QMatrix4x4, QVector4D, QWheelEvent, QMouseEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (QCheckBox, QDialog, QLabel, QPushButton, QWidget, 
                             QColorDialog, QComboBox, QVBoxLayout, QHBoxLayout,
                             QMessageBox, QScrollArea, QFrame, QSplitter,
                             QToolButton, QApplication)
from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLDebugLogger, QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture, \
    QOpenGLVersionProfile, QOpenGLVertexArrayObject, QOpenGLVersionFunctionsFactory

from DDS.DDSFile import DDSFile

from dds_sources import (
    DdsSourceSet,
    normalize_data_path,
    resolve_dds_sources,
)

if "mobase" not in sys.modules:
    import mock_mobase as mobase

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

vertexShaderCube = """
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

fragmentShaderCube = """
#version 150

uniform samplerCube aTexture;
uniform mat4 channelMatrix;
uniform vec4 channelOffset;

in vec2 texCoord;

const float PI = 3.1415926535897932384626433832795;

void main()
{
    float theta = -2.0 * PI * texCoord.x;
    float phi = PI * texCoord.y;
    gl_FragData[0] = channelMatrix * texture(aTexture, vec3(sin(theta) * sin(phi), cos(theta) * sin(phi), cos(phi))) + channelOffset;
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
    viewChanged = pyqtSignal(object)

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

        # Подключаем cleanup к destroyed виджета
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

    def viewState(self):
        return {
            "zoom": float(self.zoom),
            "pan_x": float(self.panX),
            "pan_y": float(self.panY),
        }

    def setViewState(self, state, emit_signal=False):
        if not state:
            return
        self.zoom = max(self.minZoom, min(self.maxZoom, float(state.get("zoom", self.zoom))))
        self.panX = float(state.get("pan_x", self.panX))
        self.panY = float(state.get("pan_y", self.panY))
        self._updateViewMatrix()
        self.update()
        if emit_signal:
            self.viewChanged.emit(self.viewState())

    def resetView(self, emit_signal=True):
        self.zoom = 1.0
        self.panX = 0.0
        self.panY = 0.0
        self._updateViewMatrix()
        self.update()
        if emit_signal:
            self.viewChanged.emit(self.viewState())

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
            self.viewChanged.emit(self.viewState())
        
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
            self.viewChanged.emit(self.viewState())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Reset zoom and pan on double click"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.resetView(emit_signal=True)
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
        vertexShader = vertexShader2D
        if self.ddsFile.isCubemap:
            fragmentShader = fragmentShaderCube
            vertexShader = vertexShaderCube
            if QOpenGLContext.currentContext().hasExtension(b"GL_ARB_seamless_cube_map"):
                GL_TEXTURE_CUBE_MAP_SEAMLESS = 0x884F
                gl.glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
        elif self.ddsFile.glFormat.samplerType == "F":
            fragmentShader = fragmentShaderFloat
        elif self.ddsFile.glFormat.samplerType == "UI":
            fragmentShader = fragmentShaderUInt
        else:
            fragmentShader = fragmentShaderSInt

        self.program = QOpenGLShaderProgram(self)
        self.program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vertexShader)
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

        # Очистка буферов кадра для предотвращения артефактов
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Рисуем фон (шахматку) с единичной матрицей, чтобы фон не зависел от зума/панорамирования
        self.transparecyProgram.bind()
        bgViewMatrix = QMatrix4x4()  # identity
        self.transparecyProgram.setUniformValue("viewMatrix", bgViewMatrix)
        backgroundColour = self.ddsOptions.getBackgroundColour()
        if backgroundColour and backgroundColour.isValid():
            self.transparecyProgram.setUniformValue("backgroundColour", backgroundColour)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        self.transparecyProgram.release()

        # Рисуем DDS-текстуру с текущей матрицей вида (зум/панорамирование)
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
            return  # Already cleaned up
            
        self.clean = True  # Mark as cleaned immediately to prevent re-entry
        
        try:
            context = self.context()
            if context and context.isValid():
                try:
                    self.makeCurrent()
                except RuntimeError as e:
                    print(f"DEBUG: Failed to make context current: {str(e)}")
                    # Drop references to prevent Qt from attempting to destroy them later
                    self.program = None
                    self.transparecyProgram = None
                    self.texture = None
                    self.vbo = None
                    self.vao = None
                    return

                # Release shader programs
                if hasattr(self, 'program') and self.program:
                    try:
                        self.program.release()
                    except:
                        pass
                    self.program = None

                if hasattr(self, 'transparecyProgram') and self.transparecyProgram:
                    try:
                        self.transparecyProgram.release()
                    except:
                        pass
                    self.transparecyProgram = None

                # Release texture with explicit context check
                if hasattr(self, 'texture') and self.texture:
                    try:
                        if self.context() and self.context().isValid():
                            self.texture.destroy()
                    except (RuntimeError, AttributeError) as e:
                        print(f"DEBUG: Failed to destroy texture: {str(e)}")
                    finally:
                        self.texture = None

                # Release VBO
                if hasattr(self, 'vbo') and self.vbo:
                    try:
                        self.vbo.destroy()
                    except:
                        pass
                    self.vbo = None

                # Release VAO
                if hasattr(self, 'vao') and self.vao:
                    try:
                        self.vao.destroy()
                    except:
                        pass
                    self.vao = None

                try:
                    self.doneCurrent()
                except:
                    pass
            else:
                print("DEBUG: OpenGL context is invalid or unavailable, skipping GPU resource cleanup")
                # Drop references to prevent Qt from attempting to destroy them later
                self.program = None
                self.transparecyProgram = None
                self.texture = None
                self.vbo = None
                self.vao = None
        except (RuntimeError, SystemError, AttributeError) as e:
            print(f"DEBUG: Error during cleanup: {str(e)}")

    def tr(self, str):
        return QCoreApplication.translate("DDSWidget", str)

class ColourChannels(enum.Enum):
    RGBA = "Color and Alpha"
    RGB = "Color"
    A = "Alpha"
    R = "Red"
    G = "Green"
    B = "Blue"


class DDSChannelManager:
    def __init__(self, channels: ColourChannels):
        self.channels = channels

    def setChannels(self, options: DDSOptions, channels: ColourChannels):
        self.channels = channels

        def drawColour(alpha: bool):
            colorMatrix = QMatrix4x4()
            colorOffset = QVector4D()
            if not alpha:
                colorMatrix[3, 3] = 0
                colorOffset.setW(1.0)
            options.setChannelMatrix(colorMatrix)
            options.setChannelOffset(colorOffset)

        def drawGrayscale(channel: ColourChannels):
            colorOffset = QVector4D(0, 0, 0, 1)
            channelVector = [0, 0, 0, 0]
            if channels == ColourChannels.R:
                channelVector[0] = 1
            elif channels == ColourChannels.G:
                channelVector[1] = 1
            elif channels == ColourChannels.B:
                channelVector[2] = 1
            elif channels == ColourChannels.A:
                channelVector[3] = 1
            else:
                raise ValueError("channel must be a single color channel.")
            alphaVector = [0, 0, 0, 0]
            colorMatrix = channelVector * 3 + alphaVector
            options.setChannelMatrix(colorMatrix)
            options.setChannelOffset(colorOffset)

        if channels == ColourChannels.RGBA:
            drawColour(True)
        elif channels == ColourChannels.RGB:
            drawColour(False)
        else:
            drawGrayscale(channels)


class PreviewDialogChromeGuard:
    def __init__(self, widget: QWidget):
        self.widget = widget
        self.preview_dialog = None
        self.hidden_widgets = []
        self.applied = False

    def apply(self):
        if self.applied:
            return
        self.applied = True

        parent = self.widget.parentWidget()
        while parent is not None:
            if parent.objectName() == "PreviewDialog":
                self.preview_dialog = parent
                break
            parent = parent.parentWidget()

        if self.preview_dialog is None:
            return

        for name in ("nameLabel", "modLabel", "previousButton", "nextButton"):
            child = self.preview_dialog.findChild(QWidget, name)
            if child is not None:
                self.hidden_widgets.append((child, child.isVisible()))
                child.setVisible(False)

        self.preview_dialog.destroyed.connect(self.restore)

    def restore(self):
        for child, was_visible in self.hidden_widgets:
            try:
                child.setVisible(was_visible)
            except RuntimeError:
                pass
        self.hidden_widgets.clear()


class DdsPreviewPane(QWidget):
    viewChanged = pyqtSignal(object)
    providerChanged = pyqtSignal()

    def __init__(self, title: str, options: DDSOptions, debugContext=False, parent=None):
        super().__init__(parent)
        self.options = options
        self.debugContext = debugContext
        self.sources = DdsSourceSet([], "")
        self.current_index = 0
        self.current_widget = None
        self._loading = False

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self.title_label, 1)

        self.prev_button = QToolButton()
        self.prev_button.setText("<")
        self.prev_button.setToolTip(self.tr("Previous DDS source"))
        self.prev_button.clicked.connect(lambda: self.selectRelativeProvider(-1))
        header.addWidget(self.prev_button)

        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self.selectProvider)
        header.addWidget(self.source_combo, 3)

        self.next_button = QToolButton()
        self.next_button.setText(">")
        self.next_button.setToolTip(self.tr("Next DDS source"))
        self.next_button.clicked.connect(lambda: self.selectRelativeProvider(1))
        header.addWidget(self.next_button)

        layout.addLayout(header)

        self.location_label = QLabel("")
        self.location_label.setWordWrap(True)
        self.location_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.location_label)

        self.view_host = QWidget()
        self.view_layout = QVBoxLayout(self.view_host)
        self.view_layout.setContentsMargins(0, 0, 0, 0)
        self.view_layout.setSpacing(0)
        layout.addWidget(self.view_host, 1)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.info_label)

    def setSources(self, sources: DdsSourceSet, index: int = 0):
        self.sources = sources
        if sources.providers:
            self.current_index = max(0, min(index, len(sources.providers) - 1))
        else:
            self.current_index = 0

        self._loading = True
        self.source_combo.clear()
        for provider in sources.providers:
            suffix = ""
            if provider.source_kind == "bsa":
                suffix = " (BSA)"
            elif provider.source_kind == "memory":
                suffix = " (Archive Preview)"
            self.source_combo.addItem(provider.display_name + suffix)
        self.source_combo.setCurrentIndex(self.current_index if sources.providers else -1)
        self._loading = False
        self._updateSourceControls()
        self._loadCurrentProvider()

    def currentProvider(self):
        if 0 <= self.current_index < len(self.sources.providers):
            return self.sources.providers[self.current_index]
        return None

    def currentViewState(self):
        if isinstance(self.current_widget, DDSWidget):
            return self.current_widget.viewState()
        return {}

    def setViewState(self, state, emit_signal=False):
        if isinstance(self.current_widget, DDSWidget):
            self.current_widget.setViewState(state, emit_signal=emit_signal)

    def resetView(self):
        if isinstance(self.current_widget, DDSWidget):
            self.current_widget.resetView(emit_signal=True)

    def updateRenderOptions(self):
        if isinstance(self.current_widget, DDSWidget):
            self.current_widget.update()

    def cleanup(self):
        self._setViewWidget(None)

    def selectRelativeProvider(self, delta: int):
        if not self.sources.providers:
            return
        self.selectProvider((self.current_index + delta) % len(self.sources.providers))

    def selectProvider(self, index: int):
        if self._loading:
            return
        if not (0 <= index < len(self.sources.providers)):
            return
        if index == self.current_index and isinstance(self.current_widget, DDSWidget):
            return
        view_state = self.currentViewState()
        self.current_index = index
        if self.source_combo.currentIndex() != index:
            self.source_combo.setCurrentIndex(index)
        self._loadCurrentProvider(view_state)
        self.providerChanged.emit()

    def _loadCurrentProvider(self, view_state=None):
        provider = self.currentProvider()
        if provider is None:
            self.location_label.setText(self.tr("No DDS providers found."))
            self.info_label.setText("")
            self._setViewWidget(QLabel(self.tr("No preview available.")))
            self._updateSourceControls()
            return

        self.title_label.setText(provider.filename)
        self.location_label.setText(provider.location_text)

        try:
            dds_file = self._loadDdsFile(provider)
            widget = DDSWidget(dds_file, self.options, self.debugContext)
            widget.viewChanged.connect(self.viewChanged.emit)
            self._setViewWidget(widget)
            self.info_label.setText(dds_file.getDescription())
            if view_state:
                widget.setViewState(view_state)
        except Exception as e:
            label = QLabel(self.tr(f"Error loading DDS source:\n{str(e)}"))
            label.setWordWrap(True)
            self._setViewWidget(label)
            self.info_label.setText("")

        self._updateSourceControls()

    def _loadDdsFile(self, provider):
        if provider.source_kind == "loose" and provider.physical_path:
            dds_file = DDSFile.fromFile(provider.physical_path)
        else:
            dds_file = DDSFile(provider.data, provider.virtual_path or provider.filename)
        dds_file.load()
        return dds_file

    def _setViewWidget(self, widget):
        old = self.current_widget
        if old is not None:
            self.view_layout.removeWidget(old)
            try:
                if isinstance(old, DDSWidget) and not old.clean:
                    old.cleanup()
            except Exception:
                pass
            old.setParent(None)
            old.deleteLater()

        self.current_widget = widget
        if widget is not None:
            self.view_layout.addWidget(widget)

    def _updateSourceControls(self):
        count = len(self.sources.providers)
        self.prev_button.setEnabled(count > 1)
        self.next_button.setEnabled(count > 1)
        self.source_combo.setEnabled(count > 1)

    def tr(self, str_):
        return QCoreApplication.translate("DdsPreviewPane", str_)


class DdsManageFilesDialog(QDialog):
    def __init__(self, organizer, sources: DdsSourceSet, parent=None):
        super().__init__(parent)
        self.organizer = organizer
        self.sources = sources
        self.changed = False
        self.setWindowTitle(self.tr("Manage DDS Sources"))
        self.resize(760, 520)

        layout = QVBoxLayout(self)
        help_label = QLabel(self.tr(
            "Loose overwritten DDS files can be hidden or backed up and removed. "
            "BSA and current archive preview entries are read-only."
        ))
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        self.body_layout = QVBoxLayout(body)
        self.body_layout.setSpacing(6)
        self.body_layout.setContentsMargins(6, 6, 6, 6)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        close_btn = QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

        self._populate()

    def _populate(self):
        for index, provider in enumerate(self.sources.providers):
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            row = QVBoxLayout(frame)
            row.setContentsMargins(8, 6, 8, 6)

            title = QLabel(f"#{index + 1}. {provider.display_name}")
            title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(title)

            details = QLabel(
                f"Kind: {provider.source_kind}\n"
                f"Path: {provider.location_text}"
            )
            details.setWordWrap(True)
            details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(details)

            actions = QHBoxLayout()
            actions.addStretch()
            eligible = provider.is_manageable_loose_file and index != self.sources.current_index

            if eligible:
                hide_btn = QPushButton(self.tr("Hide DDS"))
                hide_btn.clicked.connect(lambda _=False, p=provider, f=frame: self._hideProvider(p, f))
                actions.addWidget(hide_btn)

                delete_btn = QPushButton(self.tr("Delete with Backup"))
                delete_btn.clicked.connect(lambda _=False, p=provider, f=frame: self._deleteProvider(p, f))
                actions.addWidget(delete_btn)
            else:
                reason = self.tr("Read-only")
                if index == self.sources.current_index:
                    reason = self.tr("Current provider")
                elif provider.source_kind == "loose":
                    reason = self.tr("Loose source is not managed by a mod")
                readonly = QLabel(reason)
                readonly.setStyleSheet("color: #888;")
                actions.addWidget(readonly)

            row.addLayout(actions)
            self.body_layout.addWidget(frame)

        self.body_layout.addStretch()

    def _hideProvider(self, provider, frame):
        answer = QMessageBox.question(
            self,
            self.tr("Confirm Hide"),
            self.tr(
                f"Hide this DDS from mod '{provider.display_name}'?\n\n"
                f"{provider.physical_path}\n\n"
                "It will be renamed to *.mohidden and tracked for the Hidden Files Manager."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        hidden_path = provider.physical_path + ".mohidden"
        try:
            if os.path.exists(hidden_path):
                raise ValueError("Hidden target already exists.")
            shutil.move(provider.physical_path, hidden_path)
            self._updateJson(provider.owner, provider.physical_path, hidden_path)
            self._notifyModChanged(provider.owner)
            self.changed = True
            frame.setEnabled(False)
            QMessageBox.information(self, self.tr("DDS Hidden"), self.tr("DDS file hidden."))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Failed to hide DDS:\n{str(e)}"))

    def _deleteProvider(self, provider, frame):
        answer = QMessageBox.question(
            self,
            self.tr("Confirm Delete"),
            self.tr(
                f"Back up and remove this DDS from mod '{provider.display_name}'?\n\n"
                f"{provider.physical_path}"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        try:
            backup_mod_name = time.strftime("DDSPreview_Backup_%Y_%m_%d_%H_%M_%S")
            backup_mod = None
            try:
                backup_mod = self.organizer.modList().getMod(backup_mod_name)
                if not backup_mod:
                    backup_mod = self.organizer.createMod(
                        mobase.GuessedString(value=backup_mod_name, quality=mobase.GuessQuality.PRESET)
                    )
            except Exception:
                backup_mod = None

            backup_root = backup_mod.absolutePath() if backup_mod else os.path.join(self.organizer.modsPath(), backup_mod_name)
            relative = provider.virtual_path or os.path.basename(provider.physical_path)
            destination = os.path.join(backup_root, provider.owner, *relative.split("/"))
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.move(provider.physical_path, destination)
            self._removeEmptyFoldersRecursive(os.path.dirname(provider.physical_path), self._modRoot(provider.owner))
            self._notifyModChanged(provider.owner)
            if backup_mod:
                try:
                    self.organizer.modDataChanged(backup_mod)
                except Exception:
                    pass
            self.changed = True
            frame.setEnabled(False)
            QMessageBox.information(
                self,
                self.tr("DDS Deleted"),
                self.tr(f"DDS moved to backup mod:\n{backup_mod_name}"),
            )
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Failed to delete DDS:\n{str(e)}"))

    def _modRoot(self, mod_name: str) -> str:
        try:
            mod = self.organizer.modList().getMod(mod_name)
            if mod:
                return mod.absolutePath()
        except Exception:
            pass
        return ""

    def _notifyModChanged(self, mod_name: str):
        try:
            mod = self.organizer.modList().getMod(mod_name)
            if mod:
                self.organizer.modDataChanged(mod)
        except Exception:
            pass

    def _updateJson(self, mod_name: str, original_path: str, hidden_path: str):
        root = self._modRoot(mod_name)
        if not root:
            return
        json_path = os.path.join(root, "dds_actions.json")
        data = {"hidden": []}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                data = {"hidden": []}
        hidden = data.setdefault("hidden", [])
        hidden.append({"original": original_path, "hidden": hidden_path})
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4)

    def _removeEmptyFoldersRecursive(self, path, root):
        if not path or not root or not os.path.exists(path):
            return
        root = os.path.abspath(root)
        path = os.path.abspath(path)
        if not path.startswith(root):
            return
        for child in list(os.listdir(path)):
            child_path = os.path.join(path, child)
            if os.path.isdir(child_path):
                self._removeEmptyFoldersRecursive(child_path, root)
        try:
            if path != root and not os.listdir(path):
                os.rmdir(path)
        except Exception:
            pass

    def tr(self, str_):
        return QCoreApplication.translate("DdsManageFilesDialog", str_)


class DdsPreviewWidget(QWidget):
    SETTINGS_ORG = "xAI"
    SETTINGS_APP = "DDSPreview"
    SPLIT_KEY = "splitPreview"
    SPLITTER_KEY = "dds_splitter_sizes"

    def __init__(self, sources: DdsSourceSet, organizer, options: DDSOptions, channelManager: DDSChannelManager,
                 plugin, file_name: str = "", file_data: bytes = b"", parent=None):
        super().__init__(parent)
        self.sources = sources
        self.organizer = organizer
        self.options = options
        self.channelManager = channelManager
        self.plugin = plugin
        self.file_name = file_name
        self.file_data = file_data or b""
        self.settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        self._syncing = False
        self._chrome_guard = PreviewDialogChromeGuard(self)

        self.setMinimumWidth(900)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.reset_button = QPushButton(self.tr("Reset Camera"))
        self.reset_button.clicked.connect(self.resetCameras)
        toolbar.addWidget(self.reset_button)

        self.split_check = QCheckBox(self.tr("Split Preview"))
        self.split_check.setEnabled(len(self.sources.providers) > 1)
        split_saved = self.settings.value(self.SPLIT_KEY, False, type=bool)
        self.split_check.setChecked(bool(split_saved and len(self.sources.providers) > 1))
        self.split_check.toggled.connect(self.setSplitPreview)
        toolbar.addWidget(self.split_check)

        self.sync_check = QCheckBox(self.tr("Sync Cameras"))
        self.sync_check.setChecked(True)
        toolbar.addWidget(self.sync_check)

        self.find_refs_button = QPushButton(self.tr("Find NIF References"))
        self.find_refs_button.clicked.connect(self.findNifReferences)
        toolbar.addWidget(self.find_refs_button)

        self.manage_button = QPushButton(self.tr("Manage Files..."))
        self.manage_button.clicked.connect(self.manageFiles)
        toolbar.addWidget(self.manage_button)

        toolbar.addStretch()

        self.channel_combo = QComboBox()
        self._channel_keys = [e.name for e in ColourChannels]
        self.channel_combo.addItems([e.value for e in ColourChannels])
        self.channel_combo.setCurrentText(self.channelManager.channels.value)
        self.channel_combo.currentIndexChanged.connect(self._onChannelChanged)
        toolbar.addWidget(self.channel_combo)

        self.background_button = QPushButton(self.tr("Pick background color"))
        self.background_button.clicked.connect(self.pickBackgroundColour)
        toolbar.addWidget(self.background_button)

        layout.addLayout(toolbar)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(6)
        self.left_pane = DdsPreviewPane(
            self.tr("Active DDS"),
            self.options,
            self._logGlErrors(),
            self.splitter,
        )
        self.right_pane = DdsPreviewPane(
            self.tr("Compare DDS"),
            self.options,
            self._logGlErrors(),
            self.splitter,
        )
        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.right_pane)
        self.splitter.splitterMoved.connect(self._saveSplitterSizes)
        layout.addWidget(self.splitter, 1)

        self.footer = QLabel(self.tr(
            "Use mouse wheel to zoom, right/middle button + drag to pan, double-click to reset."
        ))
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)

        self.left_pane.viewChanged.connect(lambda state: self._onPaneViewChanged(self.left_pane, self.right_pane, state))
        self.right_pane.viewChanged.connect(lambda state: self._onPaneViewChanged(self.right_pane, self.left_pane, state))

        self._loadPanes()
        self.destroyed.connect(self._onDestroyed)

    def showEvent(self, event):
        super().showEvent(event)
        self._chrome_guard.apply()

    def closeEvent(self, event):
        self._onDestroyed()
        super().closeEvent(event)

    def resetCameras(self):
        if self.split_check.isChecked():
            self.left_pane.resetView()
            self.right_pane.resetView()
        else:
            self.left_pane.resetView()

    def setSplitPreview(self, enabled: bool):
        enabled = bool(enabled and len(self.sources.providers) > 1)
        if self.split_check.isChecked() != enabled:
            self.split_check.blockSignals(True)
            self.split_check.setChecked(enabled)
            self.split_check.blockSignals(False)
        self.settings.setValue(self.SPLIT_KEY, enabled)
        self.right_pane.setVisible(enabled)
        self.sync_check.setVisible(enabled)
        self.reset_button.setText(self.tr("Reset Cameras" if enabled else "Reset Camera"))
        if enabled and isinstance(self.left_pane.current_widget, DDSWidget):
            self.right_pane.setViewState(self.left_pane.currentViewState())
        self._restoreSplitterSizes()

    def findNifReferences(self):
        provider = self.left_pane.currentProvider() or self.sources.current_provider()
        query = ""
        if provider is not None:
            query = normalize_data_path(provider.virtual_path or provider.filename)
        else:
            query = normalize_data_path(self.sources.virtual_path or self.file_name)

        app = QApplication.instance()
        bridge = getattr(app, "ump_find_nif_references", None) if app else None
        if not callable(bridge):
            QMessageBox.information(
                self,
                self.tr("UMP Required"),
                self.tr("Find NIF References requires Ultimate Mod Panel to be installed and enabled."),
            )
            return

        try:
            handled = bool(bridge(query))
        except Exception as e:
            QMessageBox.warning(self, self.tr("UMP Error"), self.tr(f"UMP could not start the search:\n{str(e)}"))
            return

        if not handled:
            QMessageBox.information(
                self,
                self.tr("UMP Unavailable"),
                self.tr("UMP is enabled, but its NIF <-> DDS search tab is unavailable."),
            )

    def manageFiles(self):
        dialog = DdsManageFilesDialog(self.organizer, self.sources, self)
        dialog.exec()
        if dialog.changed:
            self.refreshSources()

    def refreshSources(self):
        self.sources = resolve_dds_sources(self.organizer, self.file_name, self.file_data)
        self.split_check.setEnabled(len(self.sources.providers) > 1)
        if len(self.sources.providers) < 2:
            self.split_check.setChecked(False)
        self._loadPanes()

    def pickBackgroundColour(self):
        newColour = QColorDialog.getColor(
            self.options.getBackgroundColour(),
            self,
            self.tr("Background color"),
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not newColour.isValid():
            return
        if self.options.getBackgroundColour().alpha() == 0:
            newColour.setAlpha(255)
        self.plugin.setPluginSetting("background r", newColour.red())
        self.plugin.setPluginSetting("background g", newColour.green())
        self.plugin.setPluginSetting("background b", newColour.blue())
        self.plugin.setPluginSetting("background a", newColour.alpha())
        self.options.setBackgroundColour(newColour)
        self._updatePanes()

    def _onChannelChanged(self, newIndex: int):
        if not (0 <= newIndex < len(self._channel_keys)):
            return
        channels = ColourChannels[self._channel_keys[newIndex]]
        self.channelManager.setChannels(self.options, channels)
        self.plugin.setPluginSetting("channels", self.channelManager.channels.name)
        self._updatePanes()

    def _loadPanes(self):
        current = self.sources.current_index
        self.left_pane.setSources(self.sources, current)

        compare = 0
        if len(self.sources.providers) > 1:
            compare = 1 if current == 0 else 0
        self.right_pane.setSources(self.sources, compare)
        self.setSplitPreview(self.split_check.isChecked())

    def _updatePanes(self):
        self.left_pane.updateRenderOptions()
        self.right_pane.updateRenderOptions()

    def _onPaneViewChanged(self, source: DdsPreviewPane, target: DdsPreviewPane, state):
        if self._syncing or not self.split_check.isChecked() or not self.sync_check.isChecked():
            return
        self._syncing = True
        try:
            target.setViewState(state, emit_signal=False)
        finally:
            self._syncing = False

    def _restoreSplitterSizes(self):
        values = self.settings.value(self.SPLITTER_KEY, [int(self.width() / 2), int(self.width() / 2)])
        try:
            values = [int(value) for value in values]
            if len(values) == 2 and all(value > 0 for value in values):
                self.splitter.setSizes(values)
        except Exception:
            self.splitter.setSizes([1, 1])

    def _saveSplitterSizes(self, *args):
        self.settings.setValue(self.SPLITTER_KEY, [int(value) for value in self.splitter.sizes()])

    def _logGlErrors(self):
        try:
            return bool(self.plugin.pluginSetting("log gl errors"))
        except Exception:
            return False

    def _onDestroyed(self, *args):
        try:
            self._saveSplitterSizes()
        except Exception:
            pass
        try:
            self.left_pane.cleanup()
            self.right_pane.cleanup()
        except Exception:
            pass
        self._chrome_guard.restore()

    def tr(self, str_):
        return QCoreApplication.translate("DdsPreviewWidget", str_)


class DDSPreview(mobase.IPluginPreview):
    def __init__(self):
        super().__init__()
        self.__organizer = None
        self.options = None
        self.channelManager = None
        self.active_widgets = []

    def __del__(self):
        self._cleanupWidgets()
    
    def _cleanupWidgets(self):
        for widget_ref in self.active_widgets[:]:
            widget = widget_ref() if isinstance(widget_ref, weakref.ref) else widget_ref
            if widget is not None:
                try:
                    if isinstance(widget, DDSWidget) and not widget.clean:
                        widget.cleanup()
                    widget.deleteLater()
                except (RuntimeError, SystemError, AttributeError) as e:
                    print(f"DEBUG: Error cleaning up widget: {str(e)}")
        self.active_widgets.clear()

    def init(self, organizer):
        print(f"DEBUG: Initializing plugin, organizer = {organizer}")
        self.__organizer = organizer
        savedColour = QColor(
            self.pluginSetting("background r"),
            self.pluginSetting("background g"),
            self.pluginSetting("background b"),
            self.pluginSetting("background a")
        )
        try:
            savedChannels = ColourChannels[self.pluginSetting("channels")]
        except KeyError:
            savedChannels = ColourChannels.RGBA
        self.options = DDSOptions(savedColour)
        self.channelManager = DDSChannelManager(savedChannels)
        self.channelManager.setChannels(self.options, savedChannels)
        return True

    def pluginSetting(self, name):
        return self.__organizer.pluginSetting(self.name(), name)

    def setPluginSetting(self, name, value):
        self.__organizer.setPluginSetting(self.name(), name, value)

    def name(self):
        return "DDS Preview Plugin"

    def author(self):
        return "AnyOldName3"

    def description(self):
        return self.tr("NIF-style DDS preview with split comparison, source cycling, "
                      "BSA-aware conflict providers, and UMP NIF reference handoff.")

    def version(self):
        return mobase.VersionInfo(2, 0, 0, 0)

    def settings(self):
        return [
            mobase.PluginSetting("log gl errors", 
                self.tr("If enabled, log OpenGL errors and debug messages. May decrease performance."), 
                False),
            mobase.PluginSetting("background r", 
                self.tr("Red channel of background color"), 0),
            mobase.PluginSetting("background g", 
                self.tr("Green channel of background color"), 0),
            mobase.PluginSetting("background b", 
                self.tr("Blue channel of background color"), 0),
            mobase.PluginSetting("background a", 
                self.tr("Alpha channel of background color"), 0),
            mobase.PluginSetting("channels", 
                self.tr("The color channels that are displayed."),
                ColourChannels.RGBA.name)
        ]

    def supportedExtensions(self):
        return {"dds"}

    def supportsArchives(self) -> bool:
        return True

    def __makePreviewWidget(self, source_set: DdsSourceSet, file_name: str, file_data: bytes = b"") -> QWidget:
        if not source_set.providers:
            error_widget = QLabel(self.tr("No DDS providers found for this preview."))
            error_widget.setWordWrap(True)
            return error_widget

        widget = DdsPreviewWidget(
            source_set,
            self.__organizer,
            self.options,
            self.channelManager,
            self,
            file_name,
            file_data,
        )
        self.active_widgets.append(weakref.ref(widget))

        def cleanup_on_destroy(*args):
            self.active_widgets = [
                ref for ref in self.active_widgets
                if (ref() if isinstance(ref, weakref.ref) else ref) is not None
            ]

        widget.destroyed.connect(cleanup_on_destroy)
        return widget

    def genFilePreview(self, fileName: str, maxSize: QSize) -> QWidget:
        print(f"DEBUG: genFilePreview called, organizer = {self.__organizer}")
        print(f"DEBUG: fileName = {fileName}, type = {type(fileName)}")

        if not fileName or not isinstance(fileName, str):
            error_widget = QLabel(self.tr(f"Error: Invalid file path provided: {fileName}"))
            error_widget.setWordWrap(True)
            return error_widget

        source_set = resolve_dds_sources(self.__organizer, fileName, b"")
        return self.__makePreviewWidget(source_set, fileName, b"")

    def genDataPreview(self, fileData: bytes, fileName: str, maxSize: QSize) -> QWidget:
        data = bytes(fileData or b"")
        print(f"DEBUG: genDataPreview called, fileName = {fileName}, data length = {len(data)}")
        source_set = resolve_dds_sources(self.__organizer, fileName, data)
        return self.__makePreviewWidget(source_set, fileName, data)

    def tr(self, str):
        return QCoreApplication.translate("DDSPreview", str)

def createPlugin():
    return DDSPreview()

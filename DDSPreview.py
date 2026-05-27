import struct
import sys
import threading
import enum
import os
import pathlib
import shutil
import time
import weakref
import json
import subprocess

from PyQt6.QtCore import QCoreApplication, qDebug, Qt, QPoint, QSize, QSettings
from PyQt6.QtGui import QColor, QOpenGLContext, QSurfaceFormat, QMatrix4x4, QVector4D, QIcon, QWheelEvent, QMouseEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (QCheckBox, QDialog, QGridLayout, QLabel, QPushButton, QWidget, 
                             QColorDialog, QComboBox, QListWidget, QListWidgetItem, QVBoxLayout,
                             QHBoxLayout, QMessageBox, QGroupBox, QScrollArea, QFrame, QSplitter)
from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLDebugLogger, QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture, \
    QOpenGLVersionProfile, QOpenGLVertexArrayObject, QOpenGLFunctions_4_1_Core, QOpenGLVersionFunctionsFactory

from DDS.DDSFile import DDSFile

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
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self.isPanning = True
            self.lastMousePos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Stop panning"""
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
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
            elif channel == ColourChannels.G:
                channelVector[1] = 1
            elif channel == ColourChannels.B:
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


class ConflictInfo:
    def __init__(self, organizer, file_path):
        self.organizer = organizer
        self.file_path = file_path
        self.conflicts = []
        self.hidden_files = {}  # mod_name: list of {'original': path, 'hidden': path.mohidden}
        self.nif_addiction_data = None  # Данные из NifDDsaddiction.json
        self.bsa_conflicts = []  # BSA конфликты для текущего DDS
        self.nif_references = []  # NIF файлы ссылающиеся на текущий DDS
        self.analysis_loaded = False  # Флаг: данные анализа из JSON загружены
        self._loadConflicts()

    def loadAnalysisData(self):
        """Опционально загружает данные анализа из JSON и находит BSA/NIF ссылки."""
        self._loadNifAddictionData()
        self.bsa_conflicts = []
        self.nif_references = []
        self._findBSAConflicts()
        self._findNifReferences()
        self.analysis_loaded = True

    def _loadNifAddictionData(self):
        """Получает кешированные данные из NifDDsaddiction.json через DDSPreview"""
        try:
            # Используем кешированные данные из DDSPreview
            self.nif_addiction_data = DDSPreview.getNifAddictionData(self.organizer)
        except Exception as e:
            # В случае ошибки используем пустые данные
            self.nif_addiction_data = None

    def _normalize_path(self, path):
        """Нормализует путь для сравнения"""
        return path.replace('\\', '/').lower()

    def _extract_filename(self, path):
        """Извлекает имя файла из пути"""
        return os.path.basename(path).lower()

    def _findBSAConflicts(self):
        """Находит BSA конфликты для текущего DDS файла"""
        if not self.nif_addiction_data:
            return
            
        current_filename = self._extract_filename(self.file_path)
        
        for mod_data in self.nif_addiction_data:
            mod_name = mod_data.get('mod_name', '')
            dds_files = mod_data.get('dds_files', [])
            
            for dds_path in dds_files:
                # Проверяем только BSA файлы
                if '.bsa]/' in dds_path:
                    dds_filename = self._extract_filename(dds_path)
                    if dds_filename == current_filename:
                        self.bsa_conflicts.append({
                            'mod_name': mod_name,
                            'path': dds_path,
                            'type': 'BSA'
                        })

    def _findNifReferences(self):
        """Находит NIF файлы ссылающиеся на текущий DDS"""
        if not self.nif_addiction_data:
            return
            
        current_filename = self._extract_filename(self.file_path)
        current_path_normalized = self._normalize_path(self.file_path)
        
        for mod_data in self.nif_addiction_data:
            mod_name = mod_data.get('mod_name', '')
            nif_files = mod_data.get('nif_files', {})
            
            for nif_path, dds_refs in nif_files.items():
                for dds_ref in dds_refs:
                    dds_ref_filename = self._extract_filename(dds_ref)
                    dds_ref_normalized = self._normalize_path(dds_ref)
                    
                    # Проверяем совпадение по имени файла или полному пути
                    if (dds_ref_filename == current_filename or 
                        current_path_normalized in dds_ref_normalized):
                        self.nif_references.append({
                            'mod_name': mod_name,
                            'nif_path': nif_path,
                            'dds_ref': dds_ref
                        })

    def _loadConflicts(self):
        mods_directory = self.organizer.modsPath()
        try:
            relative_path = os.path.join(*pathlib.Path(self.file_path).relative_to(mods_directory).parts[1:])
        except (ValueError, IndexError):
            return
        
        origins = self.organizer.getFileOrigins(relative_path)
        for origin in origins:
            mod = self.organizer.modList().getMod(origin)
            if mod:
                mod_path = mod.absolutePath()
                full_path = os.path.join(mod_path, relative_path)
                
                # Load JSON for this mod
                json_path = os.path.join(mod_path, 'dds_actions.json')
                hidden_in_mod = []
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            hidden_in_mod = data.get('hidden', [])
                            self.hidden_files[origin] = hidden_in_mod
                    except Exception as e:
                        print(f"DEBUG: Error loading JSON for mod {origin}: {str(e)}")
                
                # Check if this file is hidden
                is_hidden = False
                for h in hidden_in_mod:
                    if h['hidden'] == full_path + '.mohidden':
                        full_path = h['hidden']  # Use hidden path for size, etc.
                        is_hidden = True
                        break
                
                if os.path.exists(full_path) or is_hidden:
                    file_size = os.path.getsize(full_path) if os.path.exists(full_path) else 0
                    is_current = os.path.normpath(full_path) == os.path.normpath(self.file_path)
                    self.conflicts.append({
                        'mod_name': origin,
                        'mod': mod,
                        'path': full_path,
                        'relative_path': relative_path,
                        'size': file_size,
                        'is_current': is_current,
                        'is_hidden': is_hidden
                    })

    def getActiveFile(self):
        return self.conflicts[0] if self.conflicts else None
    
    def getOverwrittenFiles(self):
        return self.conflicts[1:] if len(self.conflicts) > 1 else []


class ConflictWidget(QWidget):
    def __init__(self, organizer, conflict_info, options, parent=None):
        super().__init__(parent)
        self.organizer = organizer
        self.conflict_info = conflict_info
        self.options = options
        self.deleted_files = {}
        self.preview_widgets = []
        self._initUI()
    
    def __del__(self):
        self._cleanupPreviews()
    
    def _cleanupPreviews(self):
        """Properly clean up all preview widgets"""
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
        
        # Создаем вертикальный сплиттер: сверху конфликты, снизу NIF ссылки
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Верхняя часть - конфликты (70%) в том же столбце
        conflicts_widget = QWidget()
        conflicts_layout = QVBoxLayout(conflicts_widget)
        
        # Панель инструментов для опционального JSON-сканирования
        toolbar_layout = QHBoxLayout()
        self.scan_btn = QPushButton(self.tr("Load Inf"))
        self.scan_btn.setToolTip(self.tr("Download analysis from JSON. Without clicking, only data from the MO2 API is shown."))
        self.scan_btn.clicked.connect(self._onScanClicked)
        toolbar_layout.addWidget(self.scan_btn)
        toolbar_layout.addStretch()
        conflicts_layout.addLayout(toolbar_layout)
        
        # Обернуть весь контент конфликтов в QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        
        active_group = QGroupBox(self.tr("Active File (Wins Conflict)"))
        active_layout = QVBoxLayout()
        
        active_file = self.conflict_info.getActiveFile()
        if active_file:
            active_widget = self._createFileWidget(active_file, 0, is_active=True)
            active_layout.addWidget(active_widget)
        else:
            active_layout.addWidget(QLabel(self.tr("File not found in mods")))
        
        active_group.setLayout(active_layout)
        content_layout.addWidget(active_group)
        
        overwritten = self.conflict_info.getOverwrittenFiles()
        if overwritten:
            overwritten_group = QGroupBox(self.tr(f"Overwritten Files ({len(overwritten)})"))
            overwritten_layout = QVBoxLayout()
            
            overwritten_scroll = QScrollArea()
            overwritten_scroll.setWidgetResizable(True)
            overwritten_content = QWidget()
            overwritten_scroll_layout = QVBoxLayout(overwritten_content)
            
            for i, file_info in enumerate(overwritten):
                file_widget = self._createFileWidget(file_info, i + 1, is_active=False)
                overwritten_scroll_layout.addWidget(file_widget)
            
            overwritten_scroll_layout.addStretch()
            overwritten_scroll.setWidget(overwritten_content)
            overwritten_layout.addWidget(overwritten_scroll, 1)  # Stretch to fill space
            
            overwritten_group.setLayout(overwritten_layout)
            content_layout.addWidget(overwritten_group, 1)  # Stretch to fill space
        else:
            no_conflicts = QLabel(self.tr("No conflicts - this file doesn't override other files"))
            content_layout.addWidget(no_conflicts)
        
        # Добавляем BSA конфликты (опционально, по кнопке Scan)
        # Создаем секцию и наполняем ее в _refreshBsaSection
        self.bsa_section = QWidget()
        self.bsa_section.setLayout(QVBoxLayout())
        content_layout.addWidget(self.bsa_section)
        
        # Обновляем содержимое секции (покажет заглушку до загрузки анализа)
        self._refreshBsaSection()
        
        content_layout.addStretch()
        scroll.setWidget(scroll_content)
        conflicts_layout.addWidget(scroll, 1)  # Верхняя часть сплиттера
        
        # Нижняя часть - NIF файлы (опционально, по кнопке Scan) в том же столбце, со своим скроллом
        nif_widget = QWidget()
        nif_layout = QVBoxLayout(nif_widget)
        
        # Секция NIF ссылок, наполняется в _refreshNifSection
        self.nif_section = QWidget()
        self.nif_section.setLayout(QVBoxLayout())
        nif_layout.addWidget(self.nif_section, 1)
        
        # Обновляем содержимое секции (покажет заглушку до загрузки анализа)
        self._refreshNifSection()
        
        # Добавляем виджеты в вертикальный сплиттер (тот же столбец)
        splitter.addWidget(conflicts_widget)
        splitter.addWidget(nif_widget)
        
        # Сохраняем ссылку на сплиттер и восстанавливаем размеры из настроек
        self._splitter = splitter
        settings = QSettings("xAI", "DDSPreview")
        sizes = settings.value("conflicts_nif_splitter_sizes", [700, 300])
        try:
            sizes = [int(s) for s in sizes]
        except (TypeError, ValueError):
            sizes = [700, 300]
        if not isinstance(sizes, (list, tuple)) or len(sizes) != 2:
            sizes = [700, 300]
        splitter.setSizes(sizes)

        # Устанавливаем пропорции: 70% для конфликтов (сверху), 30% для NIF (снизу)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        
        # Сохраняем размеры при уничтожении виджета
        def save_splitter_sizes():
            settings.setValue("conflicts_nif_splitter_sizes", [int(s) for s in splitter.sizes()])
        self.destroyed.connect(save_splitter_sizes)
        
        main_layout.addWidget(splitter, 1)
        self.setLayout(main_layout)
    
    def _createFileWidget(self, file_info, index, is_active=False):
        widget = QFrame()
        widget.setFrameShape(QFrame.Shape.StyledPanel)
        main_layout = QVBoxLayout()
        
        top_layout = QHBoxLayout()
        
        prefix = "🏆 " if is_active else f"#{index}. "
        
        # Get relative path from mods directory
        try:
            mods_path = self.organizer.modsPath()
            full_path = file_info['path']
            relative_display_path = os.path.relpath(full_path, mods_path)
        except:
            relative_display_path = file_info['path']
        
        info_text = f"<b>{prefix}{file_info['mod_name']}</b><br>" \
                   f"Size: {self._formatSize(file_info['size'])}<br>" \
                   f"Path: {relative_display_path}"
        
        if file_info['is_current']:
            info_text = f"<span style='color: #00AA00;'>➤ CURRENT FILE</span><br>" + info_text
        
        if file_info.get('is_hidden', False):
            info_text += "<br><span style='color: #2196f3;'>🔒 (Hidden)</span>"
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top_layout.addWidget(info_label, 1)
        
        btn_layout = QVBoxLayout()
        
        preview_btn = QPushButton(self.tr("Preview"))
        preview_btn.setToolTip(self.tr("Show preview of this file"))
        preview_btn.clicked.connect(lambda: self._showPreview(file_info, widget))
        btn_layout.addWidget(preview_btn)
        
        if not is_active:
            if not file_info.get('is_hidden', False):
                hide_btn = QPushButton(self.tr("Hide DDS"))
                hide_btn.setToolTip(self.tr("Hide this file by renaming to *.dds.mohidden"))
                hide_btn.clicked.connect(lambda: self._hideFile(file_info, widget))
                btn_layout.addWidget(hide_btn)
                widget.hide_btn = hide_btn
            
            delete_btn = QPushButton(self.tr("Delete"))
            delete_btn.setToolTip(self.tr("Delete this file (backup will be created)"))
            delete_btn.clicked.connect(lambda: self._deleteFile(file_info, widget))
            btn_layout.addWidget(delete_btn)
            widget.delete_btn = delete_btn
        
        top_layout.addLayout(btn_layout)
        main_layout.addLayout(top_layout)
        
        preview_container = QWidget()
        preview_layout = QVBoxLayout()
        preview_container.setLayout(preview_layout)
        preview_container.setVisible(False)
        main_layout.addWidget(preview_container)
        
        widget.setLayout(main_layout)
        
        
        widget.file_info = file_info
        widget.preview_container = preview_container
        widget.preview_layout = preview_layout
        widget.preview_btn = preview_btn
        
        return widget

    def _createBSAWidget(self, bsa_info, index):
        """Создает виджет для отображения BSA конфликта"""
        widget = QFrame()
        widget.setFrameShape(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout()
        
        info_text = f"<b>#{index}. {bsa_info['mod_name']} (BSA)</b><br>" \
                   f"Path: {bsa_info['path']}"
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(info_label)
        
        widget.setLayout(layout)
        return widget

    def _createNifWidget(self, nif_info, index):
        """Создает виджет для отображения NIF файла ссылающегося на DDS"""
        widget = QFrame()
        widget.setFrameShape(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout()
        
        info_text = f"<b>#{index}. {nif_info['mod_name']}</b><br>" \
                   f"NIF: {nif_info['nif_path']}<br>" \
                   f"DDS Reference: {nif_info['dds_ref']}"
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(info_label)
        
        widget.setLayout(layout)
        return widget
    
    def _clearLayout(self, layout):
        """Удаляет все виджеты из переданного layout"""
        try:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    try:
                        w.deleteLater()
                    except Exception:
                        pass
        except Exception:
            pass
    
    def _refreshBsaSection(self):
        """Обновляет секцию BSA конфликтов"""
        layout = self.bsa_section.layout()
        self._clearLayout(layout)
        
        if not getattr(self.conflict_info, 'analysis_loaded', False):
            lbl = QLabel(self.tr("The analysis is not loaded. Click 'Loaad Inf'."))
            layout.addWidget(lbl)
            return
        
        bsa_conflicts = self.conflict_info.bsa_conflicts or []
        if not bsa_conflicts:
            lbl = QLabel(self.tr("BSA archives were not found according to the analysis."))
            layout.addWidget(lbl)
            return
        
        group = QGroupBox(self.tr(f"BSA Archives ({len(bsa_conflicts)})"))
        g_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        s_layout = QVBoxLayout(content)
        
        for i, bsa_info in enumerate(bsa_conflicts):
            bsa_widget = self._createBSAWidget(bsa_info, i + 1)
            s_layout.addWidget(bsa_widget)
        
        s_layout.addStretch()
        scroll.setWidget(content)
        g_layout.addWidget(scroll, 1)
        group.setLayout(g_layout)
        layout.addWidget(group)
    
    def _refreshNifSection(self):
        """Обновляет секцию NIF ссылок"""
        layout = self.nif_section.layout()
        self._clearLayout(layout)
        
        if not getattr(self.conflict_info, 'analysis_loaded', False):
            lbl = QLabel(self.tr("To load data, click the button 'Load Inf'."))
            layout.addWidget(lbl)
            return
        
        nif_refs = self.conflict_info.nif_references or []
        if not nif_refs:
            lbl = QLabel(self.tr("There are no NIF files referencing this DDS based on the analysis."))
            layout.addWidget(lbl)
            return
        
        group = QGroupBox(self.tr(f"NIF Files Referencing This DDS ({len(nif_refs)})"))
        g_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        s_layout = QVBoxLayout(content)
        
        for i, nif_info in enumerate(nif_refs):
            ref_widget = self._createNifWidget(nif_info, i + 1)
            s_layout.addWidget(ref_widget)
        
        s_layout.addStretch()
        scroll.setWidget(content)
        g_layout.addWidget(scroll, 1)
        group.setLayout(g_layout)
        layout.addWidget(group)
    
    def _onScanClicked(self):
        """Обработчик кнопки Scan: загружает JSON анализ и обновляет UI"""
        if hasattr(self, 'scan_btn'):
            self.scan_btn.setEnabled(False)
            self.scan_btn.setText(self.tr("Scanning..."))
        
        # Временные заглушки "Загрузка..."
        for section in (getattr(self, 'bsa_section', None), getattr(self, 'nif_section', None)):
            if section is not None:
                lay = section.layout()
                self._clearLayout(lay)
                lbl = QLabel(self.tr("Загрузка анализа из JSON..."))
                lay.addWidget(lbl)
        
        try:
            # Загружаем анализ (использует кеш)
            self.conflict_info.loadAnalysisData()
        except Exception as e:
            # Показываем ошибку
            for section in (getattr(self, 'bsa_section', None), getattr(self, 'nif_section', None)):
                if section is not None:
                    lay = section.layout()
                    self._clearLayout(lay)
                    err = QLabel(self.tr(f"Ошибка загрузки анализа: {str(e)}"))
                    lay.addWidget(err)
            if hasattr(self, 'scan_btn'):
                self.scan_btn.setEnabled(True)
                self.scan_btn.setText(self.tr("Load Inf"))
            return
        
        # Обновляем секции с загруженными данными
        self._refreshBsaSection()
        self._refreshNifSection()
        
        if hasattr(self, 'scan_btn'):
            self.scan_btn.setEnabled(True)
            self.scan_btn.setText(self.tr("Rescan"))
    
    def _showPreview(self, file_info, parent_widget):
        """Shows or hides file preview"""
        preview_container = parent_widget.preview_container
        preview_layout = parent_widget.preview_layout
        preview_btn = parent_widget.preview_btn
        
        if preview_container.isVisible():
            # Hide preview
            preview_container.setVisible(False)
            preview_btn.setText(self.tr("Preview"))
            # Clean up content
            while preview_layout.count():
                item = preview_layout.takeAt(0)
                if item.widget():
                    widget = item.widget()
                    try:
                        if isinstance(widget, DDSWidget) and not widget.clean:
                            widget.cleanup()
                        widget.deleteLater()
                    except (RuntimeError, SystemError, AttributeError) as e:
                        print(f"DEBUG: Error cleaning up preview widget: {str(e)}")
                    # Удаляем из списка отслеживания
                    self.preview_widgets = [w for w in self.preview_widgets 
                                          if (w() if isinstance(w, weakref.ref) else w) != widget]
        else:
            # Show preview
            try:
                file_path = file_info['path']
                dds_file = DDSFile.fromFile(file_path)
                dds_file.load()
                
                dds_widget = DDSWidget(dds_file, self.options, False)
                dds_widget.setMinimumHeight(200)
                dds_widget.setMaximumHeight(300)
                
                self.preview_widgets.append(weakref.ref(dds_widget))
                
                info_label = QLabel(self.tr("💡 Tip: Use mouse wheel to zoom, right/middle click + drag to pan, double-click to reset"))
                preview_layout.addWidget(info_label)
                
                preview_layout.addWidget(dds_widget)
                preview_container.setVisible(True)
                preview_btn.setText(self.tr("Hide"))
                
            except Exception as e:
                error_label = QLabel(self.tr(f"Error loading preview: {str(e)}"))
                preview_layout.addWidget(error_label)
                preview_container.setVisible(True)
                preview_btn.setText(self.tr("Hide"))
    
    def _deleteFile(self, file_info, widget):
        answer = QMessageBox.question(
            self,
            self.tr("Confirm Deletion"),
            self.tr(f"Are you sure you want to delete the file from mod '{file_info['mod_name']}'?\n\n"
                   f"A backup will be created for recovery."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if answer != QMessageBox.StandardButton.Yes:
            return
        
        backup_mod_name = time.strftime("DDSPreview_Backup_%Y_%m_%d_%H_%M_%S")
        backup_mod = self.organizer.modList().getMod(backup_mod_name)
        if not backup_mod:
            backup_mod = self.organizer.createMod(
                mobase.GuessedString(value=backup_mod_name, quality=mobase.GuessQuality.PRESET)
            )
        
        backup_mod_path = backup_mod.absolutePath()
        src_path = file_info['path']
        dst_path = os.path.join(backup_mod_path, file_info['mod_name'], file_info['relative_path'])
        
        try:
            # Cleanup all preview widgets BEFORE filesystem changes
            self._cleanupPreviews()
            
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.move(src_path, dst_path)
            self._removeEmptyFoldersRecursive(
                os.path.dirname(src_path),
                file_info['mod'].absolutePath()
            )
            
            self.deleted_files[file_info['path']] = {
                'backup_path': dst_path,
                'backup_mod': backup_mod,
                'original_mod': file_info['mod']
            }
            
            widget.setEnabled(False)
            if hasattr(widget, 'delete_btn'):
                widget.delete_btn.setText(self.tr("Deleted"))
            
            # Notify MO2 AFTER all cleanup and UI updates
            self.organizer.modDataChanged(file_info['mod'])
            self.organizer.modDataChanged(backup_mod)
            
            QMessageBox.information(
                self,
                self.tr("File Deleted"),
                self.tr(f"File deleted and saved to backup mod:\n{backup_mod_name}")
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr(f"Failed to delete file:\n{str(e)}")
            )
    
    def _hideFile(self, file_info, widget):
        answer = QMessageBox.question(
            self,
            self.tr("Confirm Hiding"),
            self.tr(f"Are you sure you want to hide the file from mod '{file_info['mod_name']}'?\n\n"
                   f"It will be renamed to '{os.path.basename(file_info['path'])}.mohidden'\n\n"
                   f"Use 'DDS Hidden Files Manager' plugin to restore hidden files."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if answer != QMessageBox.StandardButton.Yes:
            return
        
        src_path = file_info['path']
        dst_path = src_path + '.mohidden'
        
        try:
            if os.path.exists(dst_path):
                raise ValueError("Hidden file already exists.")
            
            # Cleanup all preview widgets BEFORE filesystem changes
            self._cleanupPreviews()
            
            os.rename(src_path, dst_path)
            self._removeEmptyFoldersRecursive(
                os.path.dirname(src_path),
                file_info['mod'].absolutePath()
            )
            
            # Update JSON
            self._updateJson(file_info['mod'], 'hide', src_path, dst_path)
            
            # Update UI
            file_info['path'] = dst_path
            file_info['is_hidden'] = True
            widget.setEnabled(False)
            if hasattr(widget, 'hide_btn'):
                widget.hide_btn.setText(self.tr("Hidden"))
                widget.hide_btn.setEnabled(False)
            widget.findChild(QLabel).setText(widget.findChild(QLabel).text() + "<br>🔒 (Hidden)")
            
            # Notify MO2 AFTER all cleanup and UI updates
            self.organizer.modDataChanged(file_info['mod'])
            
            QMessageBox.information(
                self,
                self.tr("File Hidden"),
                self.tr(f"File hidden successfully!\n\n"
                       f"Use 'DDS Hidden Files Manager' plugin (Tools menu) to restore it.")
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr(f"Failed to hide file:\n{str(e)}")
            )
    
    def _restoreFile(self, file_info, widget):
        # This method is kept for future use but not exposed in UI
        pass
    
    def _updateJson(self, mod, action, original_path, hidden_path):
        json_path = os.path.join(mod.absolutePath(), 'dds_actions.json')
        data = {'hidden': []}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
        
        hidden_list = data['hidden']
        
        if action == 'hide':
            hidden_list.append({'original': original_path, 'hidden': hidden_path})
        elif action == 'restore':
            hidden_list = [h for h in hidden_list if h['hidden'] != hidden_path]
        
        data['hidden'] = hidden_list
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def _removeEmptyFoldersRecursive(self, path, root):
        if not os.path.exists(path):
            return
        files = os.listdir(path)
        for file_ in files:
            child_path = os.path.join(path, file_)
            if os.path.isdir(child_path):
                self._removeEmptyFoldersRecursive(child_path, root)
        files = os.listdir(path)
        if len(files) == 0 and path != root:
            try:
                os.rmdir(path)
            except:
                pass
    
    def _formatSize(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
    
    def tr(self, str_):
        return QCoreApplication.translate("ConflictWidget", str_)


class DDSPreview(mobase.IPluginPreview):
    # Статические переменные для кеширования NifDDsaddiction.json
    _nif_addiction_cache = None
    _nif_addiction_cache_mtime = None
    _nif_addiction_cache_path = None
    
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

    @classmethod
    def getNifAddictionData(cls, organizer=None):
        """
        Получает данные NifDDsaddiction.json с кешированием.
        Загружает файл только при первом обращении или если файл изменился.
        """
        import os
        import json
        
        # Определяем путь к JSON: сначала в папке плагина, затем в папке модов MO2
        plugin_json_path = os.path.join(os.path.dirname(__file__), "NifDDsaddiction.json")
        candidates = [plugin_json_path]
        if organizer:
            candidates.append(os.path.join(organizer.modsPath(), "NifDDsaddiction.json"))
        
        json_path = None
        for p in candidates:
            if os.path.exists(p):
                json_path = p
                break
        
        if json_path is None:
            return {}
        
        try:
            
            # Получаем время модификации файла
            current_mtime = os.path.getmtime(json_path)
            
            # Проверяем, нужно ли обновить кеш
            if (cls._nif_addiction_cache is None or 
                cls._nif_addiction_cache_path != json_path or
                cls._nif_addiction_cache_mtime != current_mtime):
                
                # Загружаем данные из файла
                with open(json_path, 'r', encoding='utf-8') as f:
                    cls._nif_addiction_cache = json.load(f)
                
                # Обновляем метаданные кеша
                cls._nif_addiction_cache_mtime = current_mtime
                cls._nif_addiction_cache_path = json_path
            
            return cls._nif_addiction_cache
            
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
        except Exception:
            return {}

    def name(self):
        return "DDS Preview Plugin"

    def author(self):
        return "AnyOldName3"

    def description(self):
        return self.tr("Lets you preview DDS files by uploading them to the GPU. "
                      "Shows file conflicts and allows conflict management. "
                      "Use mouse wheel to zoom, right/middle click + drag to pan.")

    def version(self):
        return mobase.VersionInfo(1, 3, 0, 0)

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
                ColourChannels.RGBA.name),
            mobase.PluginSetting("show conflicts", 
                self.tr("Show file conflicts information"), 
                True)
        ]

    def supportedExtensions(self):
        return {"dds"}

    def supportsArchives(self) -> bool:
        return True

    def genFilePreview(self, fileName: str, maxSize: QSize) -> QWidget:
        print(f"DEBUG: genFilePreview called, organizer = {self.__organizer}")
        print(f"DEBUG: fileName = {fileName}, type = {type(fileName)}")
    
        if not fileName or not isinstance(fileName, str):
            error_widget = QLabel(self.tr(f"Error: Invalid file path provided: {fileName}"))
            error_widget.setWordWrap(True)
            return error_widget
    
        try:
            ddsFile = DDSFile.fromFile(fileName)
            ddsFile.load()
        except Exception as e:
            error_widget = QLabel(self.tr(f"Error loading DDS file:\n{str(e)}"))
            error_widget.setWordWrap(True)
            return error_widget
    
        # Use QSplitter instead of QHBoxLayout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
    
        # Left column: conflicts
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
    
        # Right column: preview
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        image_layout = QGridLayout()
        image_layout.setRowStretch(0, 1)
        image_layout.setColumnStretch(0, 1)
    
        ddsWidget = DDSWidget(ddsFile, self.options, 
                             self.__organizer.pluginSetting(self.name(), "log gl errors") if self.__organizer else False)
    
        self.active_widgets.append(weakref.ref(ddsWidget))
    
        image_layout.addWidget(ddsWidget, 0, 0, 1, 3)
    
        zoom_info = QLabel(self.tr("💡 Use mouse wheel to zoom, right/middle button + drag to pan, double-click to reset"))
        zoom_info.setWordWrap(True)
        image_layout.addWidget(zoom_info, 1, 0, 1, 3)
    
        image_layout.addWidget(self.__makeLabel(ddsFile), 2, 0, 1, 1)
        image_layout.addWidget(self.__makeChannelsButton(ddsWidget), 2, 1, 1, 1)
        image_layout.addWidget(self.__makeColourButton(ddsWidget), 2, 2, 1, 1)
        image_layout.addWidget(self.__makeDeepScanButton(), 2, 3, 1, 1)
        image_layout.addWidget(self.__makeDeepScanButton(), 2, 3, 1, 1)
    
        right_layout.addLayout(image_layout)
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
    
        # Add conflicts to left column
        if self.__organizer and self.__organizer.pluginSetting(self.name(), "show conflicts"):
            conflict_info = ConflictInfo(self.__organizer, fileName)
            if conflict_info.conflicts:
                conflict_widget = ConflictWidget(self.__organizer, conflict_info, self.options)
                left_layout.addWidget(conflict_widget)
    
        # Restore splitter sizes from settings
        settings = QSettings("xAI", "DDSPreview")
        splitter_sizes = settings.value("dds_preview_splitter_sizes", [300, 600])
        try:
            # Ensure sizes are integers
            splitter_sizes = [int(size) for size in splitter_sizes]
        except (TypeError, ValueError) as e:
            print(f"DEBUG: Invalid splitter sizes {splitter_sizes}, using default: {e}")
            splitter_sizes = [300, 600]  # Fallback to default
        splitter.setSizes(splitter_sizes)
    
        widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        widget.setLayout(main_layout)
        widget.setMinimumWidth(900)
    
        # Save splitter sizes on destroy
        def save_splitter_sizes():
            settings.setValue("dds_preview_splitter_sizes", [int(size) for size in splitter.sizes()])
    
        widget.destroyed.connect(save_splitter_sizes)
    
        def cleanup_on_destroy():
            self.active_widgets = [w for w in self.active_widgets 
                                  if (w() if isinstance(w, weakref.ref) else w) is not None]
    
        widget.destroyed.connect(cleanup_on_destroy)
    
        return widget

    def genDataPreview(self, fileData: bytes, fileName: str, maxSize: QSize) -> QWidget:
        print(f"DEBUG: genDataPreview called, fileName = {fileName}, data length = {len(fileData)}")
    
        try:
            ddsFile = DDSFile(fileData, fileName)
            ddsFile.load()
        except Exception as e:
            error_widget = QLabel(self.tr(f"Error loading DDS file:\n{str(e)}"))
            error_widget.setWordWrap(True)
            return error_widget
    
        # Use QSplitter instead of QHBoxLayout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
    
        # Left column: conflicts
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
    
        # Right column: preview
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        image_layout = QGridLayout()
        image_layout.setRowStretch(0, 1)
        image_layout.setColumnStretch(0, 1)
    
        ddsWidget = DDSWidget(ddsFile, self.options, 
                             self.__organizer.pluginSetting(self.name(), "log gl errors") if self.__organizer else False)
    
        self.active_widgets.append(weakref.ref(ddsWidget))
    
        image_layout.addWidget(ddsWidget, 0, 0, 1, 3)
    
        zoom_info = QLabel(self.tr("💡 Use mouse wheel to zoom, right/middle button + drag to pan, double-click to reset"))
        zoom_info.setWordWrap(True)
        image_layout.addWidget(zoom_info, 1, 0, 1, 3)
    
        image_layout.addWidget(self.__makeLabel(ddsFile), 2, 0, 1, 1)
        image_layout.addWidget(self.__makeChannelsButton(ddsWidget), 2, 1, 1, 1)
        image_layout.addWidget(self.__makeColourButton(ddsWidget), 2, 2, 1, 1)
    
        right_layout.addLayout(image_layout)
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
    
        # Add conflicts to left column
        if self.__organizer and self.__organizer.pluginSetting(self.name(), "show conflicts"):
            conflict_info = ConflictInfo(self.__organizer, fileName)
            if conflict_info.conflicts:
                conflict_widget = ConflictWidget(self.__organizer, conflict_info, self.options)
                left_layout.addWidget(conflict_widget)
    
        # Restore splitter sizes from settings
        settings = QSettings("xAI", "DDSPreview")
        splitter_sizes = settings.value("dds_preview_splitter_sizes", [300, 600])
        try:
            # Ensure sizes are integers
            splitter_sizes = [int(size) for size in splitter_sizes]
        except (TypeError, ValueError) as e:
            print(f"DEBUG: Invalid splitter sizes {splitter_sizes}, using default: {e}")
            splitter_sizes = [300, 600]  # Fallback to default
        splitter.setSizes(splitter_sizes)
    
        widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        widget.setLayout(main_layout)
        widget.setMinimumWidth(900)
    
        # Save splitter sizes on destroy
        def save_splitter_sizes():
            settings.setValue("dds_preview_splitter_sizes", [int(size) for size in splitter.sizes()])
    
        widget.destroyed.connect(save_splitter_sizes)
    
        def cleanup_on_destroy():
            self.active_widgets = [w for w in self.active_widgets 
                                   if (w() if isinstance(w, weakref.ref) else w) is not None]

        widget.destroyed.connect(cleanup_on_destroy)

        # Обработка размеров splitter
        try:
            splitter_sizes = [int(size) for size in splitter_sizes]
        except (TypeError, ValueError) as e:
            print(f"DEBUG: Invalid splitter sizes {splitter_sizes}, using default: {e}")
            splitter_sizes = [300, 600]  # Fallback to default

        splitter.setSizes(splitter_sizes)

        # Создание виджета
        widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        widget.setLayout(main_layout)
        widget.setMinimumWidth(900)

        return widget

    
        # Save splitter sizes on destroy
        def save_splitter_sizes():
            settings.setValue("dds_preview_splitter_sizes", [int(size) for size in splitter.sizes()])
    
        widget.destroyed.connect(save_splitter_sizes)
    
        def cleanup_on_destroy():
            self.active_widgets = [w for w in self.active_widgets 
                                  if (w() if isinstance(w, weakref.ref) else w) is not None]
    
        widget.destroyed.connect(cleanup_on_destroy)
    
        return widget

    def tr(self, str):
        return QCoreApplication.translate("DDSPreview", str)

    def __makeLabel(self, ddsFile):
        label = QLabel(ddsFile.getDescription())
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def __makeColourButton(self, ddsWidget):
        button = QPushButton(self.tr("Pick background color"))

        def pickColour(unused):
            newColour = QColorDialog.getColor(
                self.options.getBackgroundColour(), 
                button, 
                "Background color",
                QColorDialog.ColorDialogOption.ShowAlphaChannel
            )
            if newColour.isValid():
                # Если текущий фон полностью прозрачный, принудительно выставим альфу 255,
                # чтобы выбранный цвет был видим.
                current_bg = self.options.getBackgroundColour()
                if current_bg and current_bg.alpha() == 0:
                    newColour.setAlpha(255)

                # Сохраняем настройки плагина
                self.setPluginSetting("background r", newColour.red())
                self.setPluginSetting("background g", newColour.green())
                self.setPluginSetting("background b", newColour.blue())
                self.setPluginSetting("background a", newColour.alpha())

                # Применяем цвет в опциях
                self.options.setBackgroundColour(newColour)

                # Обновляем конкретный виджет
                try:
                    ddsWidget.update()
                except Exception:
                    pass

                # Обновляем все активные DDS-виджеты, если они отслеживаются
                for widget_ref in getattr(self, 'active_widgets', []):
                    widget = widget_ref() if isinstance(widget_ref, weakref.ref) else widget_ref
                    if widget is not None:
                        try:
                            widget.update()
                        except Exception:
                            pass

        button.clicked.connect(pickColour)
        return button

    def __makeChannelsButton(self, ddsWidget):
        listwidget = QComboBox()
        channelKeys = [e.name for e in ColourChannels]
        channelNames = [e.value for e in ColourChannels]

        listwidget.addItems(channelNames)
        listwidget.setCurrentText(self.channelManager.channels.value)
        listwidget.setToolTip(self.tr("Select what color channels are displayed."))

        listwidget.showEvent = lambda _: listwidget.setCurrentText(self.channelManager.channels.value)

        def onChanged(newIndex):
            self.channelManager.setChannels(self.options, ColourChannels[channelKeys[newIndex]])
            self.setPluginSetting("channels", self.channelManager.channels.name)
            ddsWidget.update()

        listwidget.currentIndexChanged.connect(onChanged)
        return listwidget

    def __runDeepScan(self):
        """Запускает nifddsparser.exe с путем к папке модов"""
        try:
            # Получаем путь к папке плагинов
            plugin_folder = os.path.dirname(os.path.abspath(__file__))
            nifddsparser_path = os.path.join(plugin_folder, "nifddsparser.exe")
            
            # Получаем путь к папке модов через MO2 API
            mods_path = self.__organizer.modsPath()
            
            # Проверяем существование nifddsparser.exe
            if not os.path.exists(nifddsparser_path):
                QMessageBox.warning(None, self.tr("Error"), 
                                  self.tr(f"nifddsparser.exe not found at: {nifddsparser_path}"))
                return
            
            # Запускаем nifddsparser.exe с путем к модам как аргументом
            subprocess.Popen([nifddsparser_path, mods_path], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
            
        except Exception as e:
             QMessageBox.critical(None, self.tr("Error"), 
                                self.tr(f"Failed to run Deep Scan: {str(e)}"))

    def __makeDeepScanButton(self):
        """Создает кнопку Deep Scan для запуска nifddsparser.exe"""
        button = QPushButton(self.tr("Deep Scan"))
        button.setToolTip(self.tr("Run deep scan analysis using nifddsparser.exe"))
        button.clicked.connect(self.__runDeepScan)
        return button


def createPlugin():
    return DDSPreview()
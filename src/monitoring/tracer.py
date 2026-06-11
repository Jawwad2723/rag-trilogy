from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

def configure_tracer(service_name: str = "ask-my-docs") -> trace.Tracer:
    """Configures and initializes OpenTelemetry Tracer Provider with OTLP exporter."""
    provider = TracerProvider()
    
    otel_endpoint = settings.monitoring.otel_endpoint
    
    if otel_endpoint:
        try:
            logger.info("configuring_otlp_span_exporter", endpoint=otel_endpoint)
            # Configure OTLP gRPC/HTTP span exporter
            exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
            span_processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(span_processor)
        except Exception as e:
            logger.warning("otlp_exporter_failed_falling_back_to_console", error=str(e))
            # Fall back to console
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
    else:
        logger.info("no_otel_endpoint_provided_using_console_exporter")
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))

    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)

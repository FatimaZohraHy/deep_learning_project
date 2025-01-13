import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatRAGComponent } from './chat-rag.component';

describe('ChatRAGComponent', () => {
  let component: ChatRAGComponent;
  let fixture: ComponentFixture<ChatRAGComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatRAGComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ChatRAGComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
